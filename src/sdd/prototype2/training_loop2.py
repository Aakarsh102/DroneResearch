import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
import pandas as pd
import numpy as np 
import math
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

def process_drone_location_worker(args):
    """Worker function for processing drone location data"""
    drone_root, loc = args
    if loc == 'hyang':
        return {}
        
    obs = {}
    loc_path = os.path.join(drone_root, loc)
    if not os.path.isdir(loc_path):
        return {}
    
    for dr in os.listdir(loc_path):
        dr_path = os.path.join(loc_path, dr)
        if not os.path.isdir(dr_path):
            continue
            
        for fname in os.listdir(dr_path):
            if not fname.endswith("_annotations.txt"):
                continue
                
            vid = fname.replace("_annotations.txt", "")
            key = (loc, vid)
            
            try:
                df = pd.read_csv(os.path.join(dr_path, fname), sep=' ', header=None,
                               names=['trackId','xmin','ymin','xmax','ymax',
                                     'frame','lost','occluded','generated','label'],
                               usecols=['trackId', 'frame'])  # Only load needed columns
                
                obs[key] = set(zip(df['trackId'], df['frame'].astype(int)))
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    
    return obs

def process_video_worker(args):
    """Worker function for processing video data"""
    loc, vid, obs_sets, orig_root, classes, cls2idx, T_past, T_future, frame_subsample = args
    orig_file = os.path.join(orig_root, "annotations", loc, vid, "annotations.txt")
    
    if not os.path.isfile(orig_file):
        return []
    
    indices = []
    key = (loc, vid)
    seen = obs_sets.get(key, set())
    
    try:
        # Load only necessary data
        df = pd.read_csv(orig_file, sep=' ', header=None,
                       names=['trackId','xmin','ymin','xmax','ymax',
                             'frame','lost','occluded','generated','label'])
        
        # Filter by classes early
        df = df[df['label'].isin(classes)]
        
        if len(df) == 0:
            return []
        
        # Vectorized center calculation
        df['x'] = (df['xmin'] + df['xmax']) * 0.5
        df['y'] = (df['ymin'] + df['ymax']) * 0.5
        
        # Process each track
        for tid, grp in df.groupby('trackId'):
            grp = grp.sort_values('frame').reset_index(drop=True)
            frames = grp['frame'].values
            
            # Vectorized observation mask calculation
            obs_mask = np.array([1.0 if (tid, int(f)) in seen else 0.0 for f in frames])
            
            L = len(frames)
            window = T_past + T_future
            
            # NEW: Frame subsampling logic
            # Generate all possible starting frame offsets (0 to frame_subsample-1)
            for frame_offset in range(frame_subsample):
                for i in range(L - (window - 1) * frame_subsample - frame_offset):
                    # Calculate the subsampled frame indices
                    subsampled_indices = []
                    subsampled_frames = []
                    
                    for j in range(window):
                        target_idx = i + frame_offset + j * frame_subsample
                        if target_idx >= L:
                            break
                        subsampled_indices.append(target_idx)
                        subsampled_frames.append(frames[target_idx])
                    
                    # Check if we have enough frames for the full window
                    if len(subsampled_indices) < window:
                        continue
                    
                    # Check frame continuity with subsampling
                    expected_frame_diff = frame_subsample
                    valid_sequence = True
                    
                    for j in range(1, len(subsampled_frames)):
                        actual_diff = subsampled_frames[j] - subsampled_frames[j-1]
                        # Allow some tolerance for missing frames
                        if abs(actual_diff - expected_frame_diff) > frame_subsample // 2:
                            valid_sequence = False
                            break
                    
                    if not valid_sequence:
                        continue
                    
                    # Check observation requirement for past frames
                    past_indices = subsampled_indices[:T_past]
                    if obs_mask[past_indices].sum() < 2:
                        continue
                    
                    # Store sample index instead of full data
                    sample_idx = {
                        'location': loc,
                        'video': vid,
                        'track_id': tid,
                        'start_idx': i,
                        'frame_offset': frame_offset,
                        'subsampled_indices': subsampled_indices,
                        'subsampled_frames': subsampled_frames,
                        'window_size': window,
                        'label': cls2idx[grp.iloc[subsampled_indices[T_past - 1]]['label']]
                    }
                    
                    indices.append(sample_idx)
                
    except Exception as e:
        print(f"Error processing {loc}/{vid}: {e}")
        
    return indices

def process_location_stats_worker(args):
    """Worker function for computing location statistics"""
    orig_root, loc = args
    if loc in ['hyang', '.DS_Store']:
        return {}
    
    loc_stats = {}
    loc_path = Path(orig_root) / "annotations" / loc
    
    if not loc_path.is_dir():
        return {}
    
    for vid_path in loc_path.iterdir():
        if not vid_path.is_dir():
            continue
            
        orig_file = vid_path / "annotations.txt"
        if not orig_file.exists():
            continue
        
        video_key = f"{loc}_{vid_path.name}"
        
        try:
            # Use faster CSV reading
            df = pd.read_csv(orig_file, sep=' ', header=None, 
                           names=['trackId','xmin','ymin','xmax','ymax',
                                 'frame','lost','occluded','generated','label'],
                           usecols=['xmin','ymin','xmax','ymax'])  # Only load needed columns
            
            if len(df) > 0:
                # Vectorized center calculation
                x_centers = (df['xmin'] + df['xmax']) * 0.5
                y_centers = (df['ymin'] + df['ymax']) * 0.5
                coords = np.column_stack([x_centers, y_centers])
                
                loc_stats[video_key] = {
                    'mean': coords.mean(axis=0),
                    'std': coords.std(axis=0),
                    'min': coords.min(axis=0),
                    'max': coords.max(axis=0)
                }
        except Exception as e:
            print(f"Error processing {video_key}: {e}")
    
    return loc_stats

class OptimizedMultiAgentSequenceDataset(Dataset):
    def __init__(self, drone_data_root, original_dataset_root, classes,
                 T_past=10, T_future=10, use_deltas=True, normalize_positions=True,
                 cache_dir="dataset_cache", lazy_loading=True, num_workers=4,
                 max_agents=20, min_frames_per_agent=2, pad_value=-999.0,
                 frame_subsample=12):  # NEW: Frame subsampling parameter
        self.drone_data_root = drone_data_root
        self.orig_root = original_dataset_root
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        self.T_past = T_past
        self.T_future = T_future
        self.use_deltas = use_deltas
        self.normalize_positions = normalize_positions
        self.lazy_loading = lazy_loading
        self.num_workers = num_workers
        self.max_agents = max_agents
        self.min_frames_per_agent = self.T_past
        self.pad_value = pad_value  # Value used for padding missing agents/frames
        self.frame_subsample = frame_subsample  # NEW: Subsample every Nth frame
        self.obs_sets = None  # Will be populated during dataset building

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok = True)

        self._initialize_dataset()

    def _initialize_dataset(self):
        cache_info_file = self.cache_dir / "dataset_info.json"
        
        # Create cache signature based on parameters
        cache_signature = {
            'T_past': self.T_past,
            'T_future': self.T_future,
            'use_deltas': self.use_deltas,
            'normalize_positions': self.normalize_positions,
            'classes': sorted(self.classes),
            'max_agents': self.max_agents,
            'min_frames_per_agent': self.min_frames_per_agent,
            'frame_subsample': self.frame_subsample  # NEW: Include in cache signature
        }
        
        if cache_info_file.exists():
            with open(cache_info_file, 'r') as f:
                cached_info = json.load(f)
            
            if cached_info.get('signature') == cache_signature:
                print("Loading from cache...")
                self._load_from_cache(cached_info)
                return
        
        print("Building dataset from scratch...")
        self._build_dataset()
        
        # Save cache info
        cache_info = {
            'signature': cache_signature,
            'num_samples': len(self.sample_indices),
            'lazy_loading': self.lazy_loading
        }
        
        with open(cache_info_file, 'w') as f:
            json.dump(cache_info, f)

        
    def _load_from_cache(self, cache_info):
        
        if self.lazy_loading:
            indices_file = self.cache_dir/"sample_indices.pkl"
            with open(indices_file, 'rb') as f:
                self.sample_indices = pkl.load(f)

            if self.normalize_positions:
                stats_file = self.cache_dir / "video_stats.pkl"  # Fixed typo
                with open(stats_file, "rb") as f:
                    self.video_stats = pkl.load(f)
        else:
            samples_file = self.cache_dir / "all_samples.pkl"
            with open(samples_file, 'rb') as f:
                self.samples = pkl.load(f)

    def _build_dataset(self):
        """Build dataset using optimized processing"""
        print("Computing video statistics...")
        if self.normalize_positions:
            self.video_stats = self._compute_video_stats_fast()
        else:
            self.video_stats = None
        
        print("Building observation sets...")
        self.obs_sets = self._build_obs_sets_parallel()
        
        print("Processing multi-agent trajectories...")
        if self.lazy_loading:
            self.sample_indices = self._build_multi_agent_sample_indices_parallel(self.obs_sets)
            self._save_samples_to_hdf5()
        else:
            self.samples = self._build_multi_agent_samples_parallel(self.obs_sets)
            self._cache_samples()

    def _build_multi_agent_sample_indices_parallel(self, obs_sets):
        """Build sample indices for multi-agent trajectories with lazy loading and frame subsampling"""
        def process_video(args):
            loc, vid, obs_sets = args
            orig_file = os.path.join(self.orig_root, "annotations", loc, vid, "annotations.txt")
            if not os.path.isfile(orig_file):
                return []
            
            indices = []
            key = (loc, vid)
            seen = obs_sets.get(key)
            try:
                df = pd.read_csv(orig_file, sep=' ', header=None,
                               names=['trackId','xmin','ymin','xmax','ymax',
                                     'frame','lost','occluded','generated','label'])
                df = df[df['label'].isin(self.classes)]
                if len(df) == 0:
                    return []
                df['x'] = (df['xmin'] + df['xmax']) * 0.5
                df['y'] = (df['ymin'] + df['ymax']) * 0.5
                
                # Get all unique frames
                all_frames = sorted(df['frame'].unique())
                window_size = self.T_past + self.T_future
                
                # NEW: Frame subsampling logic for multi-agent
                # Try different starting offsets for subsampling
                for frame_offset in range(self.frame_subsample):
                    # Calculate how many subsampled windows we can fit
                    max_subsampled_windows = (len(all_frames) - frame_offset) // self.frame_subsample
                    
                    for start_subsample_idx in range(max_subsampled_windows - window_size + 1):
                        # Calculate the actual frame indices we'll use
                        window_frame_indices = []
                        window_frames = []
                        
                        for j in range(window_size):
                            frame_idx = frame_offset + (start_subsample_idx + j) * self.frame_subsample
                            if frame_idx < len(all_frames):
                                window_frame_indices.append(frame_idx)
                                window_frames.append(all_frames[frame_idx])
                        
                        if len(window_frames) < window_size:
                            continue

                        past_frames = window_frames[:self.T_past]
                        future_frames = window_frames[self.T_past:]

                        window_df = df[df['frame'].isin(window_frames)]

                        if len(window_frames) == 0:
                            continue

                        # Find agents that meet minimum frame requirement
                        valid_agents = []
                        agent_data = {}

                        for tid, agent_df in window_df.groupby('trackId'):
                            agent_frames = set(agent_df['frame'].values)
                            past_frames_count = len([f for f in past_frames if f in agent_frames])
                            future_frames_count = len([f for f in future_frames if f in agent_frames])

                            if past_frames_count >= self.min_frames_per_agent:
                                obs_mask = np.array([1.0 if (tid, int(f)) in seen else 0.0 
                                                   for f in past_frames if f in agent_frames])
                                if obs_mask.sum() >= 2:
                                    valid_agents.append(tid)
                                    agent_data[tid] = {
                                        'past_frames': past_frames_count,
                                        'future_frames': future_frames_count,
                                        'obs_count': obs_mask.sum()
                                    }

                        if len(valid_agents) == 0:
                            continue
                        
                        # Create sample index
                        sample_idx = {
                            'location': loc,
                            'video': vid,
                            'start_frame': window_frames[0],
                            'end_frame': window_frames[-1],
                            'window_frames': window_frames,
                            'frame_offset': frame_offset,  # NEW: Store frame offset
                            'subsample_start_idx': start_subsample_idx,  # NEW: Store subsample start
                            'valid_agents': valid_agents,
                            'agent_data': agent_data,
                            'num_agents': len(valid_agents)
                        }

                        indices.append(sample_idx)
            except Exception as e:
                print(f"Error processing {loc}/{vid}: {e}")
            return indices

        args = []
        for loc in os.listdir(os.path.join(self.orig_root, "annotations")):
            if loc in ['hyang', '.DS_Store']:
                continue
            loc_path = os.path.join(self.orig_root, "annotations", loc)
            if not os.path.isdir(loc_path):
                continue
            for vid in os.listdir(loc_path):
                args.append((loc, vid, obs_sets))
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_video, args))
        
        # Flatten results
        all_indices = []
        for result in results:
            all_indices.extend(result)
        
        return all_indices

    def _save_samples_to_hdf5(self):
        """Save processed multi-agent trajectory data to HDF5"""
        hdf5_file = self.cache_dir / "multi_agent_trajectory_data.h5"
        
        # Group samples by video for efficient storage
        video_groups = {}
        for idx, sample_idx in enumerate(self.sample_indices):
            video_key = f"{sample_idx['location']}_{sample_idx['video']}"
            if video_key not in video_groups:
                video_groups[video_key] = []
            video_groups[video_key].append((idx, sample_idx))
        
        with h5py.File(hdf5_file, 'w') as f:
            for video_key, samples in video_groups.items():
                if not samples:
                    continue
                
                # Load video data once
                loc, vid = video_key.split('_', 1)
                orig_file = os.path.join(self.orig_root, "annotations", loc, vid, "annotations.txt")
                
                if not os.path.isfile(orig_file):
                    continue
                
                df = pd.read_csv(orig_file, sep=' ', header=None,
                               names=['trackId','xmin','ymin','xmax','ymax',
                                     'frame','lost','occluded','generated','label'])
                df = df[df['label'].isin(self.classes)]
                df['x'] = (df['xmin'] + df['xmax']) * 0.5
                df['y'] = (df['ymin'] + df['ymax']) * 0.5
                
                # Create HDF5 group for this video
                video_grp = f.create_group(video_key)
                
                # Store trajectory data grouped by track
                track_data = {}
                for tid, grp in df.groupby('trackId'):
                    grp = grp.sort_values('frame').reset_index(drop=True)
                    track_data[tid] = {
                        'coords': grp[['x', 'y']].values.astype(np.float32),
                        'frames': grp['frame'].values.astype(np.int32),
                        'occluded': grp['occluded'].values.astype(np.float32),
                        'labels': grp['label'].values
                    }
                
                # Store track data
                for tid, data in track_data.items():
                    track_grp = video_grp.create_group(str(tid))
                    track_grp.create_dataset('coords', data=data['coords'])
                    track_grp.create_dataset('frames', data=data['frames'])
                    track_grp.create_dataset('occluded', data=data['occluded'])
                    track_grp.create_dataset('labels', data=data['labels'].astype('S10'))
        
        # Cache sample indices and stats
        with open(self.cache_dir / "sample_indices.pkl", 'wb') as f:
            pkl.dump(self.sample_indices, f)
        
        if self.video_stats:
            with open(self.cache_dir / "video_stats.pkl", 'wb') as f:
                pkl.dump(self.video_stats, f)

    def _compute_video_stats_fast(self):
        locations = [d for d in os.listdir(Path(self.orig_root) / "annotations") 
                    if d not in ['hyang', '.DS_Store']]
        
        args = [(self.orig_root, loc) for loc in locations]
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_location_stats_worker, args))
        
        # Merge results
        video_stats = {}
        for result in results:
            video_stats.update(result)
        
        return video_stats
    
    def _build_obs_sets_parallel(self) -> Dict:
        """Build observation sets using parallel processing"""
        locations = [d for d in os.listdir(self.drone_data_root) 
                    if os.path.isdir(os.path.join(self.drone_data_root, d))]
        args = [(self.drone_data_root, loc) for loc in locations]
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_drone_location_worker, args))

        merged_obs = {}
        for result in results:
            merged_obs.update(result)

        return merged_obs

    def _cache_samples(self):
        """Cache all samples to disk"""
        with open(self.cache_dir / "all_samples.pkl", 'wb') as f:
            pkl.dump(self.samples, f)

    @lru_cache(maxsize=128)
    def _load_trajectory_data(self, video_key: str, track_id: int):
        hdf5_file = self.cache_dir / "multi_agent_trajectory_data.h5"

        with h5py.File(hdf5_file, 'r') as f:  # Fixed: use h5py.File instead of open
            if video_key not in f:
                return None
            track_grp = f[video_key][str(track_id)]

            return {
                'coords': track_grp['coords'][:].astype(np.float32),
                'frames': track_grp['frames'][:].astype(np.int32),
                'occluded': track_grp['occluded'][:].astype(np.float32),
                'labels': [label.decode('utf-8') for label in track_grp['labels'][:]]
            }
        
    def _normalize_coords(self, coords, video_key):
        """Normalize coordinates using video-specific statistics"""
        if self.video_stats is None or video_key not in self.video_stats:
            return coords
        
        stats = self.video_stats[video_key]
        normalized = (coords - stats['mean']) / (stats['std'] + 1e-8)
        return normalized
    
    def get_video_stats(self, video_key):
        """Return video-specific statistics for denormalization"""
        if self.video_stats is None:
            return None
        return self.video_stats.get(video_key, None)
    
    def __len__(self):
        if self.lazy_loading:
            return len(self.sample_indices)
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        if not self.lazy_loading:
            # Use pre-loaded samples
            s = self.samples[idx]
            return self._format_sample(s)
        
        # Lazy loading: load sample on demand
        sample_idx = self.sample_indices[idx]
        return self._load_multi_agent_sample_on_demand(sample_idx)

    def _load_multi_agent_sample_on_demand(self, sample_idx):
        """Load multi-agent sample on demand from HDF5 with frame subsampling"""
        video_key = f"{sample_idx['location']}_{sample_idx['video']}"
        window_frames = sample_idx['window_frames']  # These are already subsampled frames
        valid_agents = sample_idx['valid_agents']
        
        past_frames = window_frames[:self.T_past]
        future_frames = window_frames[self.T_past:]
        
        # Initialize arrays for all agents (padded to max_agents)
        num_agents = min(len(valid_agents), self.max_agents)
        
        # Shape: (max_agents, T_past, 2), (max_agents, T_future, 2)
        # For agent padding (not enough agents), use pad_value
        past_positions = np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32)
        future_positions = np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32)
        past_positions_orig = np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32)
        future_positions_orig = np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32)
        
        # Masks: 1 = valid/present, 0 = invalid/absent
        obs_masks = np.zeros((self.max_agents, self.T_past), dtype=np.float32)  # 1 if observed by drone
        temporal_masks_past = np.zeros((self.max_agents, self.T_past), dtype=np.float32)  # 1 if agent exists at this time
        temporal_masks_future = np.zeros((self.max_agents, self.T_future), dtype=np.float32)  # 1 if agent exists at this time
        occ_masks = np.zeros((self.max_agents, self.T_future), dtype=np.float32)  # occlusion values (only valid where temporal_mask=1)
        agent_masks = np.zeros(self.max_agents, dtype=np.float32)  # 1 if agent slot is used, 0 if padded
        
        # Labels and metadata
        agent_labels = np.full(self.max_agents, -1, dtype=np.int32)
        agent_ids = np.full(self.max_agents, -1, dtype=np.int32)
        
        # Get observation set for this video
        obs_set = self.obs_sets.get((sample_idx['location'], sample_idx['video']), set())
        
        # Process each valid agent
        for agent_idx, track_id in enumerate(valid_agents[:self.max_agents]):
            track_data = self._load_trajectory_data(video_key, track_id)
            
            if track_data is None:
                continue
            
            # Mark this agent slot as used (not padded)
            agent_masks[agent_idx] = 1.0
            agent_ids[agent_idx] = track_id
            
            # Create frame to index mapping for this track
            frame_to_idx = {frame: idx for idx, frame in enumerate(track_data['frames'])}
            
            # Fill past positions (using subsampled frames)
            for t, frame in enumerate(past_frames):
                if frame in frame_to_idx:
                    data_idx = frame_to_idx[frame]
                    coords = track_data['coords'][data_idx]
                    
                    # Always use ground truth coordinates (even if not observed)
                    past_positions_orig[agent_idx, t] = coords
                    past_positions[agent_idx, t] = coords
                    
                    # Mark temporal presence
                    temporal_masks_past[agent_idx, t] = 1.0
                    
                    # Set observation mask (1 if observed by drone, 0 if GT but not observed)
                    obs_masks[agent_idx, t] = 1.0 if (track_id, int(frame)) in obs_set else 0.0
                    
                    # Get label from first valid frame
                    if agent_labels[agent_idx] == -1:
                        label = track_data['labels'][data_idx]
                        if label in self.cls2idx:
                            agent_labels[agent_idx] = self.cls2idx[label]
            
            # Fill future positions (using subsampled frames)
            for t, frame in enumerate(future_frames):
                if frame in frame_to_idx:
                    data_idx = frame_to_idx[frame]
                    coords = track_data['coords'][data_idx]
                    
                    # Always use ground truth coordinates
                    future_positions_orig[agent_idx, t] = coords
                    future_positions[agent_idx, t] = coords
                    
                    # Mark temporal presence
                    temporal_masks_future[agent_idx, t] = 1.0
                    
                    # Set occlusion (only meaningful where agent is present)
                    occ_masks[agent_idx, t] = track_data['occluded'][data_idx]
        
        
        # Normalize if requested (only normalize valid positions, not padding)
        if self.normalize_positions:
            for agent_idx in range(num_agents):
                if agent_masks[agent_idx] > 0:
                    # Normalize past positions where agent is temporally present
                    for t in range(self.T_past):
                        if temporal_masks_past[agent_idx, t] > 0:
                            past_positions[agent_idx, t] = self._normalize_coords(
                                past_positions[agent_idx, t:t+1], video_key)[0]
                    
                    # Normalize future positions where agent is temporally present  
                    for t in range(self.T_future):
                        if temporal_masks_future[agent_idx, t] > 0:
                            future_positions[agent_idx, t] = self._normalize_coords(
                                future_positions[agent_idx, t:t+1], video_key)[0]
        
        sample = {
            'past_positions': past_positions,
            'future_positions': future_positions,
            'past_positions_orig': past_positions_orig,
            'future_positions_orig': future_positions_orig,
            'obs_masks': obs_masks,  # Whether observed by drone (only meaningful in past)
            'temporal_masks_past': temporal_masks_past,  # Whether agent exists at this timestep
            'temporal_masks_future': temporal_masks_future,  # Whether agent exists at this timestep
            'occ_masks': occ_masks,  # Occlusion values (only meaningful where temporal_mask=1)
            'agent_masks': agent_masks,  # Whether agent slot is used (not padding)
            'agent_labels': agent_labels,
            'agent_ids': agent_ids,
            'location': sample_idx['location'],
            'video': sample_idx['video'],
            'start_frame': sample_idx['start_frame'],
            'num_valid_agents': num_agents,
            'subsampled_frames': window_frames,  # NEW: Store the actual subsampled frame numbers
            'frame_subsample_rate': self.frame_subsample  # NEW: Store subsample rate for reference
        }
        
        # Add delta features if requested (adjusted for subsampling)
        if self.use_deltas:
            past_deltas = np.zeros((self.max_agents, self.T_past, 2), dtype=np.float32)
            future_deltas = np.zeros((self.max_agents, self.T_future, 2), dtype=np.float32)
            
            for agent_idx in range(num_agents):
                if agent_masks[agent_idx] > 0:
                    # Calculate deltas only for temporally valid positions
                    agent_past = past_positions[agent_idx]
                    agent_future = future_positions[agent_idx]
                    past_mask = temporal_masks_past[agent_idx]
                    future_mask = temporal_masks_future[agent_idx]
                    
                    # Past deltas - calculate between consecutive valid frames
                    # NOTE: With subsampling, these deltas represent movement over frame_subsample frames
                    valid_past_indices = np.where(past_mask > 0)[0]
                    if len(valid_past_indices) > 1:
                        for i in range(1, len(valid_past_indices)):
                            curr_idx = valid_past_indices[i]
                            prev_idx = valid_past_indices[i-1]
                            past_deltas[agent_idx, curr_idx] = (
                                agent_past[curr_idx] - agent_past[prev_idx]
                            )
                    
                    # Future deltas 
                    valid_future_indices = np.where(future_mask > 0)[0]
                    
                    # Transition from past to future
                    if len(valid_past_indices) > 0 and len(valid_future_indices) > 0:
                        last_past_idx = valid_past_indices[-1]
                        first_future_idx = valid_future_indices[0]
                        future_deltas[agent_idx, first_future_idx] = (
                            agent_future[first_future_idx] - agent_past[last_past_idx]
                        )
                    
                    # Future-to-future deltas
                    if len(valid_future_indices) > 1:
                        for i in range(1, len(valid_future_indices)):
                            curr_idx = valid_future_indices[i]
                            prev_idx = valid_future_indices[i-1]
                            future_deltas[agent_idx, curr_idx] = (
                                agent_future[curr_idx] - agent_future[prev_idx]
                            )
            
            sample['past_deltas'] = past_deltas
            sample['future_deltas'] = future_deltas
        
        return self._format_sample(sample)
    
    def _format_sample(self, sample):
        """Format sample for model consumption"""
        # Convert to tensors if needed
        return sample




if __name__ == "__main__":
    dataset = OptimizedMultiAgentSequenceDataset("/Users/aakarshrai/Desktop/square_stanford_data",
                                                 "/Users/aakarshrai/Desktop/stanford_data/archive",
                                                 ['Pedestrian','Biker','Skater','Cart','Car','Bus'],
                                                 10, 20, cache_dir = "new_cache", max_agents = 256)
    
    
