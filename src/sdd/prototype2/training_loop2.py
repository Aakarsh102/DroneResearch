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
from model2 import GraphInteractionModel
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


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import argparse

# Assuming your dataset and model classes are imported
# from your_dataset_module import OptimizedMultiAgentSequenceDataset
# from your_model_module import GraphInteractionModel

class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction with uncertainty and proper masking"""
    
    def __init__(self, use_deltas=True, predict_uncertainty=True, 
                 position_weight=1.0, delta_weight=1.0, uncertainty_weight=0.1):
        super().__init__()
        self.use_deltas = use_deltas
        self.predict_uncertainty = predict_uncertainty
        self.position_weight = position_weight
        self.delta_weight = delta_weight
        self.uncertainty_weight = uncertainty_weight
        
    def forward(self, predictions, batch):
        total_loss = 0.0
        loss_dict = {}
        
        # Extract masks from batch
        # temporal_masks_future: (batch, max_agents, T_future) - 1 if agent exists at timestep
        # occ_masks: (batch, max_agents, T_future) - occlusion values (only valid where agent exists)
        # agent_masks: (batch, max_agents) - 1 if agent slot is used
        
        temporal_mask = batch['temporal_masks_future']  # (batch, max_agents, T_future)
        occ_mask = batch['occ_masks']  # (batch, max_agents, T_future) 
        agent_mask = batch['agent_masks']  # (batch, max_agents)
        
        # Create visibility mask: 1 for visible (non-occluded) positions where agent exists
        # Assuming occ_mask values: 0 = visible, 1 = occluded (adjust if different)
        visibility_mask = temporal_mask * (1.0 - occ_mask)  # (batch, max_agents, T_future)
        
        # Also mask out padded agents
        agent_mask_expanded = agent_mask.unsqueeze(-1).expand_as(visibility_mask)
        valid_mask = visibility_mask * agent_mask_expanded
        
        if self.predict_uncertainty:
            # Negative log likelihood loss for positions
            if 'future_positions_mu' in predictions and 'future_positions_logvar' in predictions:
                pos_mu = predictions['future_positions_mu']  # (batch, max_agents, T_future, 2)
                pos_logvar = predictions['future_positions_logvar']  # (batch, max_agents, T_future, 2)
                pos_target = batch['future_positions']  # (batch, max_agents, T_future, 2)
                
                pos_loss = self._gaussian_nll_loss(pos_mu, pos_logvar, pos_target, valid_mask)
                loss_dict['position_loss'] = pos_loss
                total_loss += self.position_weight * pos_loss
            
            # Negative log likelihood loss for deltas
            if self.use_deltas and 'future_deltas_mu' in predictions and 'future_deltas_logvar' in predictions:
                delta_mu = predictions['future_deltas_mu']
                delta_logvar = predictions['future_deltas_logvar']
                delta_target = batch['future_deltas']
                
                delta_loss = self._gaussian_nll_loss(delta_mu, delta_logvar, delta_target, valid_mask)
                loss_dict['delta_loss'] = delta_loss
                total_loss += self.delta_weight * delta_loss
            
            # Uncertainty regularization
            if 'future_positions_logvar' in predictions:
                pos_logvar = predictions['future_positions_logvar']
                # Apply mask to only regularize visible positions
                masked_logvar = pos_logvar * valid_mask.unsqueeze(-1).expand_as(pos_logvar)
                # Regularize towards reasonable uncertainty
                uncertainty_reg = torch.mean(masked_logvar ** 2) 
                loss_dict['uncertainty_reg'] = uncertainty_reg
                total_loss += self.uncertainty_weight * uncertainty_reg
                
        else:
            # Standard MSE loss
            if 'future_positions' in predictions:
                pos_loss = self._masked_mse_loss(
                    predictions['future_positions'], batch['future_positions'], valid_mask
                )
                loss_dict['position_loss'] = pos_loss
                total_loss += self.position_weight * pos_loss
            
            if self.use_deltas and 'future_deltas' in predictions:
                delta_loss = self._masked_mse_loss(
                    predictions['future_deltas'], batch['future_deltas'], valid_mask
                )
                loss_dict['delta_loss'] = delta_loss
                total_loss += self.delta_weight * delta_loss
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
    
    def _gaussian_nll_loss(self, mu, logvar, target, mask):
        """
        Negative log likelihood for Gaussian distribution
        
        Args:
            mu: predicted mean (batch, max_agents, T_future, 2)
            logvar: predicted log variance (batch, max_agents, T_future, 2) 
            target: ground truth (batch, max_agents, T_future, 2)
            mask: validity mask (batch, max_agents, T_future) - 1 for valid, 0 for invalid
        """
        # Expand mask to match tensor dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(mu)  # (batch, max_agents, T_future, 2)
        
        # Compute NLL
        var = torch.exp(logvar)  # Convert log variance to variance
        squared_error = (target - mu) ** 2
        nll = 0.5 * (logvar + squared_error / (var + 1e-8))  # Add epsilon for stability
        
        # Apply mask - only compute loss for valid positions
        masked_nll = nll * mask_expanded
        
        # Return average loss over valid positions
        valid_count = mask_expanded.sum()
        if valid_count > 0:
            return masked_nll.sum() / valid_count
        else:
            return torch.tensor(0.0, device=mu.device, requires_grad=True)
    
    def _masked_mse_loss(self, pred, target, mask):
        """MSE loss with masking for invalid positions"""
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)
        mse = ((pred - target) ** 2) * mask_expanded
        
        valid_count = mask_expanded.sum()
        if valid_count > 0:
            return mse.sum() / valid_count
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def collate_fn(batch):
    """Custom collate function for batching"""
    # Stack all tensors
    batched = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            batched[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], np.ndarray):
            batched[key] = torch.from_numpy(np.stack([item[key] for item in batch]))
        else:
            # For non-tensor data like strings, lists, etc.
            batched[key] = [item[key] for item in batch]
    
    return batched


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_dict_accum = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            # Forward pass
            predictions = model(batch, use_teacher_forcing=False)  # No teacher forcing during validation
            
            # Compute loss
            loss, loss_dict = criterion(predictions, batch)
            
            batch_size = batch['past_positions'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in loss_dict_accum:
                    loss_dict_accum[key] = 0.0
                loss_dict_accum[key] += value.item() * batch_size
    
    # Average losses
    avg_loss = total_loss / total_samples
    for key in loss_dict_accum:
        loss_dict_accum[key] /= total_samples
    
    return avg_loss, loss_dict_accum


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_dict_accum = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        # Forward pass
        predictions = model(batch, use_teacher_forcing=True)  # Use teacher forcing during training
        
        # Compute loss
        loss, loss_dict = criterion(predictions, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (optional but recommended for sequence models)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate statistics
        batch_size = batch['past_positions'].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        for key, value in loss_dict.items():
            if key not in loss_dict_accum:
                loss_dict_accum[key] = 0.0
            loss_dict_accum[key] += value.item() * batch_size
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Average losses
    avg_loss = total_loss / total_samples
    for key in loss_dict_accum:
        loss_dict_accum[key] /= total_samples
    
    return avg_loss, loss_dict_accum


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Agent Trajectory Prediction Model')
    parser.add_argument('--drone_data_root', type=str, required=True, 
                       help='Path to drone data root directory')
    parser.add_argument('--original_dataset_root', type=str, required=True,
                       help='Path to original dataset root directory')
    parser.add_argument('--cache_dir', type=str, default='dataset_cache',
                       help='Directory for dataset cache')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--max_agents', type=int, default=128, help='Maximum number of agents')
    parser.add_argument('--T_past', type=int, default=10, help='Past time steps')
    parser.add_argument('--T_future', type=int, default=20, help='Future time steps')
    parser.add_argument('--frame_subsample', type=int, default=12, help='Frame subsampling rate')
    parser.add_argument('--wandb_project', type=str, default='trajectory-prediction',
                       help='Weights & Biases project name')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for logging')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )
    
    # Dataset configuration
    classes = ['Pedestrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus']
    locations = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'nexus', 'quad']  # Update with your actual locations
    
    print("Loading dataset...")
    dataset = OptimizedMultiAgentSequenceDataset(
        drone_data_root=args.drone_data_root,
        original_dataset_root=args.original_dataset_root,
        classes=classes,
        T_past=args.T_past,
        T_future=args.T_future,
        use_deltas=True,
        normalize_positions=True,
        cache_dir=args.cache_dir,
        lazy_loading=True,
        num_workers=args.num_workers,
        max_agents=args.max_agents,
        frame_subsample=args.frame_subsample
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    model = GraphInteractionModel(
        num_classes=len(classes),
        locations=locations,  
        d_model=256,
        nhead=8,
        num_layers=6,
        T_past=args.T_past,
        T_future=args.T_future,
        max_agents=args.max_agents
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize loss function and optimizer
    criterion = TrajectoryLoss(
        use_deltas=True,
        predict_uncertainty=True,
        position_weight=1.0,
        delta_weight=1.0,
        uncertainty_weight=0.1
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_loss_dict = evaluate_model(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Add detailed loss components
        for key, value in train_loss_dict.items():
            metrics[f'train_{key}'] = value
        for key, value in val_loss_dict.items():
            metrics[f'val_{key}'] = value
        
        wandb.log(metrics)
        
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args)
        }
        
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"New best model saved with val_loss: {val_loss:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch} epochs")
            break
        
        print("-" * 50)
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'args': vars(args)
    }
    torch.save(final_checkpoint, checkpoint_dir / 'final_model.pth')
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    wandb.finish()


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     dataset = OptimizedMultiAgentSequenceDataset("/Users/aakarshrai/Desktop/square_stanford_data",
#                                                  "/Users/aakarshrai/Desktop/stanford_data/archive",
#                                                  ['Pedestrian','Biker','Skater','Cart','Car','Bus'],
#                                                  10, 20, cache_dir = "new_cache", max_agents = 128)
    
    
