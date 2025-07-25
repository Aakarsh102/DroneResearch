import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_an import LandscapeAwareTrajectoryPredictor, compute_metrics
import math
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


# Module-level functions for multiprocessing
# Module-level functions for multiprocessing
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
    loc, vid, obs_sets, orig_root, classes, cls2idx, T_past, T_future, frame_interval = args
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
            
            # Calculate minimum frames needed for past (must be complete)
            min_past_frames = T_past * frame_interval
            
            # Slide through the track, only requiring complete past
            for i in range(L):
                # Check if we have enough frames for complete past
                past_end_idx = i + min_past_frames
                if past_end_idx > L:
                    break
                
                # Get past frame indices (must be complete/contiguous)
                past_indices = list(range(i, past_end_idx, frame_interval))
                if len(past_indices) < T_past:
                    continue
                past_indices = past_indices[:T_past]  # Take exactly T_past frames
                
                # Verify past frames are reasonably spaced and observed
                past_obs_mask = obs_mask[past_indices]
                if past_obs_mask.sum() < 2:  # Need at least 2 observations in past
                    continue
                
                # Get future frame indices (can be incomplete)
                future_start_idx = past_end_idx
                future_indices = []
                future_existence_mask = []
                
                for j in range(T_future):
                    future_frame_idx = future_start_idx + j * frame_interval
                    if future_frame_idx < L:
                        # Frame exists in track
                        future_indices.append(future_frame_idx)
                        future_existence_mask.append(1.0)
                    else:
                        # Frame doesn't exist (track ended) - use padding
                        future_indices.append(-1)  # Placeholder for non-existent frame
                        future_existence_mask.append(0.0)
                
                # We need at least some future frames, but not all
                valid_future_count = sum(1 for idx in future_indices if idx != -1)
                if valid_future_count == 0:
                    continue
                
                # Create the complete trajectory indices and masks
                all_indices = past_indices + [idx for idx in future_indices if idx != -1]
                all_frames = frames[all_indices].tolist()
                
                # For non-existent future frames, we'll handle them in the dataset loading
                future_frames = []
                for idx in future_indices:
                    if idx != -1:
                        future_frames.append(frames[idx])
                    else:
                        future_frames.append(-1)  # Placeholder
                
                # Store sample index with existence information
                sample_idx = {
                    'location': loc,
                    'video': vid,
                    'track_id': tid,
                    'past_indices': past_indices,
                    'future_indices': future_indices,  # May contain -1 for non-existent frames
                    'past_frames': frames[past_indices].tolist(),
                    'future_frames': future_frames,
                    'future_existence_mask': future_existence_mask,  # 1 for existing, 0 for non-existing
                    'frame_interval': frame_interval,
                    'label': cls2idx[grp.iloc[past_indices[-1]]['label']]  # Use last past frame for label
                }
                
                indices.append(sample_idx)
                
    except Exception as e:
        print(f"Error processing {loc}/{vid}: {e}")
        
    return indices


def process_video_worker_alternative(args):
    """Alternative version with simpler logic - less efficient but clearer"""
    loc, vid, obs_sets, orig_root, classes, cls2idx, T_past, T_future, frame_interval = args
    orig_file = os.path.join(orig_root, "annotations", loc, vid, "annotations.txt")
    
    if not os.path.isfile(orig_file):
        return []
    
    indices = []
    key = (loc, vid)
    seen = obs_sets.get(key, set())
    
    try:
        df = pd.read_csv(orig_file, sep=' ', header=None,
                       names=['trackId','xmin','ymin','xmax','ymax',
                             'frame','lost','occluded','generated','label'])
        
        df = df[df['label'].isin(classes)]
        if len(df) == 0:
            return []
        
        df['x'] = (df['xmin'] + df['xmax']) * 0.5
        df['y'] = (df['ymin'] + df['ymax']) * 0.5
        
        for tid, grp in df.groupby('trackId'):
            grp = grp.sort_values('frame').reset_index(drop=True)
            frames = grp['frame'].values
            
            obs_mask = np.array([1.0 if (tid, int(f)) in seen else 0.0 for f in frames])
            L = len(frames)
            
            # Simpler approach: slide window and check only past completeness
            for start_idx in range(0, L - T_past + 1, frame_interval):
                # Past must be complete
                past_end = start_idx + T_past
                if past_end > L:
                    break
                    
                past_indices = list(range(start_idx, past_end))
                past_obs = obs_mask[past_indices]
                
                if past_obs.sum() < 2:  # Need minimum observations
                    continue
                
                # Future can be partial
                future_indices = []
                future_existence = []
                
                for i in range(T_future):
                    future_idx = past_end + i
                    if future_idx < L:
                        future_indices.append(future_idx)
                        future_existence.append(1.0)
                    else:
                        future_indices.append(-1)  # Mark as non-existent
                        future_existence.append(0.0)
                
                # Need at least one future frame
                if sum(future_existence) == 0:
                    continue
                
                sample_idx = {
                    'location': loc,
                    'video': vid,
                    'track_id': tid,
                    'past_indices': past_indices,
                    'future_indices': future_indices,
                    'past_frames': frames[past_indices].tolist(),
                    'future_frames': [frames[i] if i != -1 else -1 for i in future_indices],
                    'future_existence_mask': future_existence,
                    'frame_interval': 1,  # Since we're not using interval sampling here
                    'label': cls2idx[grp.iloc[past_indices[-1]]['label']]
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

class OptimizedAgentSequenceDataset(Dataset):
    def __init__(self, drone_data_root, original_dataset_root, classes,
                 T_past=10, T_future=10, frame_interval=12, use_deltas=True, 
                 normalize_positions=True, cache_dir="dataset_cache", 
                 lazy_loading=True, num_workers=4):
        self.drone_data_root = drone_data_root
        self.orig_root = original_dataset_root
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        self.T_past = T_past
        self.T_future = T_future
        self.frame_interval = frame_interval
        self.use_deltas = use_deltas
        self.normalize_positions = normalize_positions
        self.lazy_loading = lazy_loading
        self.num_workers = num_workers
        
        # Cache directory for processed data
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize dataset
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        """Initialize dataset with either cached data or process from scratch"""
        cache_info_file = self.cache_dir / "dataset_info.json"
        
        # Create cache signature based on parameters
        cache_signature = {
            'T_past': self.T_past,
            'T_future': self.T_future,
            'frame_interval': self.frame_interval,
            'use_deltas': self.use_deltas,
            'normalize_positions': self.normalize_positions,
            'classes': sorted(self.classes)
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
        """Load dataset from cached files"""
        if self.lazy_loading:
            # Load only sample indices and metadata
            indices_file = self.cache_dir / "sample_indices.pkl"
            with open(indices_file, 'rb') as f:
                self.sample_indices = pkl.load(f)
                
            # Load video stats if needed
            if self.normalize_positions:
                stats_file = self.cache_dir / "video_stats.pkl"
                with open(stats_file, 'rb') as f:
                    self.video_stats = pkl.load(f)
        else:
            # Load all samples into memory
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
        obs_sets = self._build_obs_sets_parallel()
        
        print("Processing trajectories...")
        if self.lazy_loading:
            self.sample_indices = self._build_sample_indices_parallel(obs_sets)
            self._save_samples_to_hdf5()  # Save raw data to HDF5 for lazy loading
        else:
            self.samples = self._build_samples_parallel(obs_sets)
            self._cache_samples()
    
    def _compute_video_stats_fast(self) -> Dict:
        """Compute video statistics using vectorized operations"""
        # Process locations in parallel
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
        
        # Merge results
        merged_obs = {}
        for result in results:
            merged_obs.update(result)
        
        return merged_obs
    
    def _build_sample_indices_parallel(self, obs_sets) -> List[Dict]:
        """Build sample indices for lazy loading"""
        # Collect all video arguments
        args = []
        for loc in os.listdir(os.path.join(self.orig_root, "annotations")):
            if loc in ['hyang', '.DS_Store']:
                continue
            loc_path = os.path.join(self.orig_root, "annotations", loc)
            if not os.path.isdir(loc_path):
                continue
            for vid in os.listdir(loc_path):
                args.append((loc, vid, obs_sets, self.orig_root, self.classes, 
                           self.cls2idx, self.T_past, self.T_future, self.frame_interval))
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_video_worker, args))
        
        # Flatten results
        all_indices = []
        for result in results:
            all_indices.extend(result)
        
        return all_indices
    
    def _build_samples_parallel(self, obs_sets):
        """Build all samples in memory (non-lazy version)"""
        # This would be similar to _build_sample_indices_parallel but would
        # actually load and process the full sample data instead of just indices
        # For brevity, I'm not implementing this here since you're using lazy loading
        raise NotImplementedError("Non-lazy loading not implemented in this version")
    
    def _save_samples_to_hdf5(self):
        """Save processed trajectory data to HDF5 for efficient lazy loading"""
        hdf5_file = self.cache_dir / "trajectory_data.h5"
        
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
        
        # Cache other data
        with open(self.cache_dir / "sample_indices.pkl", 'wb') as f:
            pkl.dump(self.sample_indices, f)
        
        if self.video_stats:
            with open(self.cache_dir / "video_stats.pkl", 'wb') as f:
                pkl.dump(self.video_stats, f)
    
    def _cache_samples(self):
        """Cache all samples to disk"""
        with open(self.cache_dir / "all_samples.pkl", 'wb') as f:
            pkl.dump(self.samples, f)
    
    @lru_cache(maxsize=128)
    def _load_trajectory_data(self, video_key: str, track_id: int):
        """Load trajectory data from HDF5 with caching"""
        hdf5_file = self.cache_dir / "trajectory_data.h5"
        
        with h5py.File(hdf5_file, 'r') as f:
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
        return self._load_sample_on_demand(sample_idx)
    
    def _load_sample_on_demand(self, sample_idx):
        """Load individual sample on demand from HDF5"""
        video_key = f"{sample_idx['location']}_{sample_idx['video']}"
        track_data = self._load_trajectory_data(video_key, sample_idx['track_id'])
        
        if track_data is None:
            raise ValueError(f"Could not load data for {video_key}, track {sample_idx['track_id']}")
        
        coords = track_data['coords']
        frames = track_data['frames']
        occluded = track_data['occluded']
        
        # Extract past sequences (always complete)
        past_indices = sample_idx['past_indices']
        past_coords = coords[past_indices]
        past_occluded = occluded[past_indices]
        
        # Extract future sequences (may be incomplete)
        future_indices = sample_idx['future_indices']
        future_existence_mask = np.array(sample_idx['future_existence_mask'])
        
        # Handle future frames - pad non-existent ones
        future_coords = np.zeros((self.T_future, 2), dtype=np.float32)
        future_occluded = np.ones(self.T_future, dtype=np.float32)  # Default to occluded
        
        for i, idx in enumerate(future_indices):
            if idx != -1:  # Frame exists
                future_coords[i] = coords[idx]
                future_occluded[i] = occluded[idx]
            else:  # Frame doesn't exist - use last known position as placeholder
                if i > 0 and future_existence_mask[i-1] == 1:
                    # Use previous position if available
                    future_coords[i] = future_coords[i-1]
                else:
                    # Use last past position
                    future_coords[i] = past_coords[-1]
                future_occluded[i] = 1.0  # Mark as "occluded" (will be ignored by loss)
        
        # Store original coordinates
        orig_past_coords = past_coords.copy()
        orig_future_coords = future_coords.copy()
        
        # Normalize if requested
        if self.normalize_positions:
            past_coords = self._normalize_coords(past_coords, video_key)
            future_coords = self._normalize_coords(future_coords, video_key)
        
        # Build observation mask for past (from drone data)
        obs_mask = np.ones(self.T_past, dtype=np.float32)  # Simplified - you may want to use actual obs data
        
        # Create existence mask for future (combines occlusion and existence)
        # For loss computation: 1 = compute loss, 0 = ignore
        future_loss_mask = future_existence_mask * (1.0 - future_occluded)  # Visible AND existing
        
        sample = {
            'past_positions': past_coords.astype(np.float32),
            'future_positions': future_coords.astype(np.float32),
            'past_positions_orig': orig_past_coords.astype(np.float32),
            'future_positions_orig': orig_future_coords.astype(np.float32),
            'obs_mask': obs_mask,
            'occ_mask': future_loss_mask.astype(np.float32),  # Combined existence + visibility mask
            'existence_mask': future_existence_mask.astype(np.float32),  # Pure existence mask
            'visibility_mask': (1.0 - future_occluded).astype(np.float32),  # Pure visibility mask
            'label': sample_idx['label'],
            'location': sample_idx['location'],
            'video': sample_idx['video'],
            'track_id': sample_idx['track_id'],
            'start_frame': int(sample_idx['past_frames'][0]),
            'frame_interval': sample_idx['frame_interval']
        }
        
        # Add delta features if requested
        if self.use_deltas:
            # Past deltas
            past_deltas = np.diff(past_coords, axis=0)
            past_deltas = np.concatenate([np.zeros((1, 2)), past_deltas], axis=0)
            
            # Future deltas - only compute for existing frames
            future_deltas = np.zeros((self.T_future, 2), dtype=np.float32)
            
            # Transition from last past to first future (if future exists)
            if future_existence_mask[0] == 1:
                future_deltas[0] = future_coords[0] - past_coords[-1]
            
            # Subsequent future deltas
            for i in range(1, self.T_future):
                if future_existence_mask[i] == 1 and future_existence_mask[i-1] == 1:
                    future_deltas[i] = future_coords[i] - future_coords[i-1]
                # else: leave as zero (will be masked out by loss)
            
            sample['past_deltas'] = past_deltas.astype(np.float32)
            sample['future_deltas'] = future_deltas.astype(np.float32)
        
        return self._format_sample(sample)
    
    def _format_sample(self, sample):
        """Format sample for consistency - implement based on your needs"""
        return sample 

# Helper functions for model training
def denormalize_positions(normalized_coords, video_stats):
    """Convert normalized coordinates back to original scale"""
    if video_stats is None:
        return normalized_coords
    
    if isinstance(normalized_coords, torch.Tensor):
        device = normalized_coords.device
        mean = torch.tensor(video_stats['mean']).to(device)
        std = torch.tensor(video_stats['std']).to(device)
        return normalized_coords * std + mean
    else:
        return normalized_coords * video_stats['std'] + video_stats['mean']

def deltas_to_positions(start_pos, deltas, frame_interval=1):
    """Convert delta predictions back to absolute positions
    
    Args:
        start_pos: Starting position
        deltas: Delta predictions (already account for frame interval)
        frame_interval: Frame interval (for reference, deltas should already be scaled)
    """
    if isinstance(deltas, torch.Tensor):
        positions = torch.zeros_like(deltas)
        positions[0] = start_pos
        for i in range(1, len(deltas)):
            positions[i] = positions[i-1] + deltas[i]
        return positions
    else:
        positions = np.zeros_like(deltas)
        positions[0] = start_pos
        for i in range(1, len(deltas)):
            positions[i] = positions[i-1] + deltas[i]
        return positions


import torch
import torch.nn as nn

# class TrajectoryLoss(nn.Module):
#     """Loss function for trajectory prediction with uncertainty"""
    
#     def __init__(self, use_deltas=True, predict_uncertainty=True, 
#                  position_weight=1.0, delta_weight=1.0, uncertainty_weight=0.1):
#         super().__init__()
#         self.use_deltas = use_deltas
#         self.predict_uncertainty = predict_uncertainty
#         self.position_weight = position_weight
#         self.delta_weight = delta_weight
#         self.uncertainty_weight = uncertainty_weight
        
#     def forward(self, predictions, targets):
#         total_loss = 0.0
#         loss_dict = {}
        
#         # FIXED: Use the correct mask - occ_mask should be 1 for visible (non-occluded) positions
#         # The original mask was inverted - occ_mask=1 means visible, but we need to use it correctly
#         valid_mask = targets['occ_mask']  # (batch, T_future) - 1 for visible, 0 for occluded
        
#         if self.predict_uncertainty:
#             # Negative log likelihood loss for positions
#             if 'future_positions_mu' in predictions:
#                 pos_mu = predictions['future_positions_mu']
#                 pos_logvar = predictions['future_positions_logvar']
#                 pos_target = targets['future_positions']
                
#                 pos_loss = self._gaussian_nll_loss(pos_mu, pos_logvar, pos_target, valid_mask)
#                 loss_dict['position_loss'] = pos_loss
#                 total_loss += self.position_weight * pos_loss
            
#             # Negative log likelihood loss for deltas
#             if self.use_deltas and 'future_deltas_mu' in predictions:
#                 delta_mu = predictions['future_deltas_mu']
#                 delta_logvar = predictions['future_deltas_logvar']
#                 delta_target = targets['future_deltas']
                
#                 delta_loss = self._gaussian_nll_loss(delta_mu, delta_logvar, delta_target, valid_mask)
#                 loss_dict['delta_loss'] = delta_loss
#                 total_loss += self.delta_weight * delta_loss
            
#             # FIXED: Uncertainty regularization - encourage reasonable uncertainty
#             # We want to prevent both overconfident (very negative logvar) and underconfident (very positive logvar)
#             if 'future_positions_logvar' in predictions:
#                 pos_logvar = predictions['future_positions_logvar']
#                 # Apply mask to only regularize visible positions
#                 masked_logvar = pos_logvar * valid_mask.unsqueeze(-1).expand_as(pos_logvar)
#                 # Regularize towards reasonable uncertainty (not too small, not too large)
#                 # This encourages logvar to be around 0 (variance = 1)
#                 uncertainty_reg = torch.mean(masked_logvar ** 2) 
#                 loss_dict['uncertainty_reg'] = uncertainty_reg
#                 total_loss += self.uncertainty_weight * uncertainty_reg
                
#         else:
#             # Standard MSE loss
#             if 'future_positions' in predictions:
#                 pos_loss = self._masked_mse_loss(
#                     predictions['future_positions'], targets['future_positions'], valid_mask
#                 )
#                 loss_dict['position_loss'] = pos_loss
#                 total_loss += self.position_weight * pos_loss
            
#             if self.use_deltas and 'future_deltas' in predictions:
#                 delta_loss = self._masked_mse_loss(
#                     predictions['future_deltas'], targets['future_deltas'], valid_mask
#                 )
#                 loss_dict['delta_loss'] = delta_loss
#                 total_loss += self.delta_weight * delta_loss
        
#         loss_dict['total_loss'] = total_loss
#         return total_loss, loss_dict
    
#     def _gaussian_nll_loss(self, mu, logvar, target, mask):
#         """
#         Negative log likelihood for Gaussian distribution
        
#         Args:
#             mu: predicted mean (batch, T_future, 2)
#             logvar: predicted log variance (batch, T_future, 2) 
#             target: ground truth (batch, T_future, 2)
#             mask: validity mask (batch, T_future) - 1 for valid, 0 for occluded
#         """
#         # Expand mask to match tensor dimensions
#         mask_expanded = mask.unsqueeze(-1).expand_as(mu)  # (batch, T_future, 2)
        
#         # Compute NLL: -log p(target|mu, var) = 0.5 * (log(2*pi*var) + (target-mu)^2/var)
#         # Since we have logvar, this becomes: 0.5 * (log(2*pi) + logvar + (target-mu)^2/exp(logvar))
#         var = torch.exp(logvar)  # Convert log variance to variance
        
#         # NLL formula (without the constant log(2*pi) term)
#         squared_error = (target - mu) ** 2
#         nll = 0.5 * (logvar + squared_error / (var + 1e-8))  # Add small epsilon for numerical stability
        
#         # Apply mask - only compute loss for non-occluded positions
#         masked_nll = nll * mask_expanded
        
#         # Return average loss over valid positions
#         valid_count = mask_expanded.sum()
#         if valid_count > 0:
#             return masked_nll.sum() / valid_count
#         else:
#             return torch.tensor(0.0, device=mu.device, requires_grad=True)
    
#     def _masked_mse_loss(self, pred, target, mask):
#         """MSE loss with masking for occluded positions"""
#         mask_expanded = mask.unsqueeze(-1).expand_as(pred)
#         mse = ((pred - target) ** 2) * mask_expanded
        
#         valid_count = mask_expanded.sum()
#         if valid_count > 0:
#             return mse.sum() / valid_count
#         else:
#             return torch.tensor(0.0, device=pred.device, requires_grad=True)



class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction with uncertainty and multiple masks"""
    
    def __init__(self, use_deltas=True, predict_uncertainty=True, 
                 position_weight=1.0, delta_weight=1.0, uncertainty_weight=0.1,
                 use_existence_mask=True, use_visibility_mask=True):
        super().__init__()
        self.use_deltas = use_deltas
        self.predict_uncertainty = predict_uncertainty
        self.position_weight = position_weight
        self.delta_weight = delta_weight
        self.uncertainty_weight = uncertainty_weight
        self.use_existence_mask = use_existence_mask
        self.use_visibility_mask = use_visibility_mask
        
    def forward(self, predictions, targets):
        total_loss = 0.0
        loss_dict = {}
        
        # Create combined validity mask based on available masks
        valid_mask = self._create_validity_mask(targets)  # (batch, T_future)
        
        if self.predict_uncertainty:
            # Negative log likelihood loss for positions
            if 'future_positions_mu' in predictions:
                pos_mu = predictions['future_positions_mu']
                pos_logvar = predictions['future_positions_logvar']
                pos_target = targets['future_positions']
                
                pos_loss = self._gaussian_nll_loss(pos_mu, pos_logvar, pos_target, valid_mask)
                loss_dict['position_loss'] = pos_loss
                total_loss += self.position_weight * pos_loss
            
            # Negative log likelihood loss for deltas
            if self.use_deltas and 'future_deltas_mu' in predictions:
                delta_mu = predictions['future_deltas_mu']
                delta_logvar = predictions['future_deltas_logvar']
                delta_target = targets['future_deltas']
                
                delta_loss = self._gaussian_nll_loss(delta_mu, delta_logvar, delta_target, valid_mask)
                loss_dict['delta_loss'] = delta_loss
                total_loss += self.delta_weight * delta_loss
            
            # Uncertainty regularization
            if 'future_positions_logvar' in predictions:
                pos_logvar = predictions['future_positions_logvar']
                # Apply mask to only regularize valid positions
                masked_logvar = pos_logvar * valid_mask.unsqueeze(-1).expand_as(pos_logvar)
                # Count valid elements for proper averaging
                valid_count = valid_mask.unsqueeze(-1).expand_as(pos_logvar).sum()
                if valid_count > 0:
                    uncertainty_reg = (masked_logvar ** 2).sum() / valid_count
                else:
                    uncertainty_reg = torch.tensor(0.0, device=pos_logvar.device, requires_grad=True)
                loss_dict['uncertainty_reg'] = uncertainty_reg
                total_loss += self.uncertainty_weight * uncertainty_reg
                
        else:
            # Standard MSE loss
            if 'future_positions' in predictions:
                pos_loss = self._masked_mse_loss(
                    predictions['future_positions'], targets['future_positions'], valid_mask
                )
                loss_dict['position_loss'] = pos_loss
                total_loss += self.position_weight * pos_loss
            
            if self.use_deltas and 'future_deltas' in predictions:
                delta_loss = self._masked_mse_loss(
                    predictions['future_deltas'], targets['future_deltas'], valid_mask
                )
                loss_dict['delta_loss'] = delta_loss
                total_loss += self.delta_weight * delta_loss
        
        loss_dict['total_loss'] = total_loss
        loss_dict['valid_ratio'] = valid_mask.float().mean()  # Track how much data is valid
        return total_loss, loss_dict
    
    def _create_validity_mask(self, targets):
        """Create combined validity mask from available masks"""
        device = next(iter(targets.values())).device if targets else torch.device('cpu')
        
        # Start with all ones (valid by default)
        if 'occ_mask' in targets:
            # occ_mask should be the combined mask (existence AND visibility)
            valid_mask = targets['occ_mask']
        else:
            # Fallback: create mask from individual components
            batch_size = targets['future_positions'].shape[0]
            T_future = targets['future_positions'].shape[1]
            valid_mask = torch.ones(batch_size, T_future, device=device)
            
            # Apply existence mask if available and requested
            if self.use_existence_mask and 'existence_mask' in targets:
                valid_mask = valid_mask * targets['existence_mask']
            
            # Apply visibility mask if available and requested
            if self.use_visibility_mask and 'visibility_mask' in targets:
                valid_mask = valid_mask * targets['visibility_mask']
        
        return valid_mask
    
    def _gaussian_nll_loss(self, mu, logvar, target, mask):
        """
        Negative log likelihood for Gaussian distribution
        
        Args:
            mu: predicted mean (batch, T_future, 2)
            logvar: predicted log variance (batch, T_future, 2) 
            target: ground truth (batch, T_future, 2)
            mask: validity mask (batch, T_future) - 1 for valid, 0 for invalid
        """
        # Expand mask to match tensor dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(mu)  # (batch, T_future, 2)
        
        # Compute NLL: -log p(target|mu, var) = 0.5 * (log(2*pi*var) + (target-mu)^2/var)
        var = torch.exp(logvar)  # Convert log variance to variance
        
        # NLL formula (without the constant log(2*pi) term)
        squared_error = (target - mu) ** 2
        nll = 0.5 * (logvar + squared_error / (var + 1e-8))  # Add epsilon for numerical stability
        
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


def compute_metrics(predictions, targets, video_stats=None):
    """
    Compute trajectory prediction metrics (ADE, FDE, etc.)
    
    Args:
        predictions: Model predictions dict
        targets: Ground truth targets dict
        video_stats: Video statistics for denormalization (optional)
    
    Returns:
        Dict of computed metrics
    """
    metrics = {}
    
    # Get validity mask
    if 'occ_mask' in targets:
        valid_mask = targets['occ_mask']  # (batch, T_future)
    elif 'existence_mask' in targets and 'visibility_mask' in targets:
        valid_mask = targets['existence_mask'] * targets['visibility_mask']
    else:
        # Default to all valid
        batch_size, T_future = targets['future_positions'].shape[:2]
        valid_mask = torch.ones(batch_size, T_future, device=targets['future_positions'].device)
    
    # Get predicted positions (use mean if uncertainty prediction)
    if 'future_positions_mu' in predictions:
        pred_positions = predictions['future_positions_mu']
    elif 'future_positions' in predictions:
        pred_positions = predictions['future_positions']
    else:
        return metrics  # No position predictions available
    
    target_positions = targets['future_positions']
    
    # Denormalize if video stats are provided
    if video_stats is not None:
        pred_positions = denormalize_positions(pred_positions, video_stats)
        target_positions = denormalize_positions(target_positions, video_stats)
    
    # Compute displacement errors
    displacement_errors = torch.norm(pred_positions - target_positions, dim=-1)  # (batch, T_future)
    
    # Apply validity mask
    valid_mask_expanded = valid_mask  # (batch, T_future)
    masked_errors = displacement_errors * valid_mask_expanded
    
    # ADE: Average Displacement Error over all valid time steps
    valid_count_total = valid_mask_expanded.sum()
    if valid_count_total > 0:
        ade = masked_errors.sum() / valid_count_total
        metrics['ADE'] = ade.item()
    else:
        metrics['ADE'] = 0.0
    
    # FDE: Final Displacement Error (last valid time step for each sequence)
    batch_size = pred_positions.shape[0]
    fde_values = []
    
    for b in range(batch_size):
        # Find the last valid time step for this sequence
        valid_timesteps = torch.where(valid_mask_expanded[b] > 0)[0]
        if len(valid_timesteps) > 0:
            last_valid_t = valid_timesteps[-1]
            fde_values.append(displacement_errors[b, last_valid_t].item())
    
    if fde_values:
        metrics['FDE'] = np.mean(fde_values)
    else:
        metrics['FDE'] = 0.0
    
    # Additional metrics
    if valid_count_total > 0:
        # RMSE (Root Mean Square Error)
        mse = (masked_errors ** 2).sum() / valid_count_total
        metrics['RMSE'] = torch.sqrt(mse).item()
        
        # Maximum displacement error
        metrics['Max_Error'] = masked_errors.max().item()
        
        # Median displacement error (approximate using valid errors)
        valid_errors = masked_errors[valid_mask_expanded > 0]
        if len(valid_errors) > 0:
            metrics['Median_Error'] = torch.median(valid_errors).item()
    
    # Uncertainty metrics (if available)
    if 'future_positions_logvar' in predictions:
        pred_logvar = predictions['future_positions_logvar']
        pred_std = torch.exp(0.5 * pred_logvar)  # Convert logvar to std
        
        # Average predicted uncertainty over valid positions
        valid_mask_3d = valid_mask.unsqueeze(-1).expand_as(pred_std)
        valid_uncertainty_count = valid_mask_3d.sum()
        
        if valid_uncertainty_count > 0:
            avg_uncertainty = (pred_std * valid_mask_3d).sum() / valid_uncertainty_count
            metrics['Avg_Uncertainty'] = avg_uncertainty.item()
            
            # Uncertainty calibration: compare predicted uncertainty to actual errors
            # This checks if high uncertainty correlates with high errors
            uncertainty_magnitude = torch.norm(pred_std, dim=-1)  # (batch, T_future)
            error_magnitude = displacement_errors  # (batch, T_future)
            
            # Compute correlation over valid positions
            valid_positions = valid_mask_expanded > 0
            if valid_positions.sum() > 1:
                valid_uncertainty = uncertainty_magnitude[valid_positions]
                valid_errors = error_magnitude[valid_positions]
                
                # Simple correlation coefficient
                uncertainty_mean = valid_uncertainty.mean()
                error_mean = valid_errors.mean()
                
                numerator = ((valid_uncertainty - uncertainty_mean) * (valid_errors - error_mean)).sum()
                denom_u = ((valid_uncertainty - uncertainty_mean) ** 2).sum()
                denom_e = ((valid_errors - error_mean) ** 2).sum()
                
                if denom_u > 0 and denom_e > 0:
                    correlation = numerator / torch.sqrt(denom_u * denom_e)
                    metrics['Uncertainty_Error_Correlation'] = correlation.item()
    
    # Coverage metrics (what fraction of positions are valid)
    metrics['Valid_Position_Ratio'] = (valid_mask_expanded.sum() / valid_mask_expanded.numel()).item()
    
    return metrics


def denormalize_positions(normalized_coords, video_stats):
    """Convert normalized coordinates back to original scale"""
    if video_stats is None:
        return normalized_coords
    
    if isinstance(normalized_coords, torch.Tensor):
        device = normalized_coords.device
        mean = torch.tensor(video_stats['mean'], device=device)
        std = torch.tensor(video_stats['std'], device=device)
        return normalized_coords * std + mean
    else:
        return normalized_coords * video_stats['std'] + video_stats['mean']
    

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import os
import json
import matplotlib.pyplot as plt

def setup_device():
    """Setup device for training"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def create_data_loaders(drone_data_root, original_dataroot, classes, T_past=10, T_future=10, 
                        use_deltas=True, normalize_positions=True, batch_size=32, 
                        train_split=0.9, val_split=0.1, num_workers=4):
    print('Loading dataset')
    dataset = OptimizedAgentSequenceDataset(
        drone_data_root, original_dataroot, classes, T_past, T_future, 
        use_deltas=use_deltas, normalize_positions=normalize_positions
    )
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    print("total size:", total_size)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, dataset

def initialize_model(num_classes, locations, device, d_model=64, num_layers=2, T_past=10, T_future=10):
    model = LandscapeAwareTrajectoryPredictor(
        num_classes, locations, d_model=d_model, num_layers=num_layers, 
        T_past=T_past, T_future=T_future
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model

def create_optimizer_scheduler(model, learning_rate=1e-4, weight_decay=1e-5, 
                             scheduler_type='cosine', num_epochs=100):
    """Create optimizer and learning rate scheduler"""
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    else:
        scheduler = None
    
    return optimizer, scheduler

def train_epoch(model, train_loader, optimizer, loss_fn, device, use_teacher_forcing=True):
    model.train()
    epoch_losses = defaultdict(list) 
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad()

        try:
            predictions = model(batch, use_teacher_forcing=use_teacher_forcing)
            loss, loss_dict = loss_fn(predictions, batch)
            
            # Check for NaN/inf losses
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss in batch {batch_idx}: {loss}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Record losses
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    epoch_losses[key].append(value.item())
                else:
                    epoch_losses[key].append(value)

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Valid%': f"{loss_dict.get('valid_ratio', 0.0):.2f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            # Skip this batch and continue
            continue
    
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    return avg_losses

def validate_epoch(model, val_loader, loss_fn, device, dataset=None):
    """Validate for one epoch"""
    model.eval()
    epoch_losses = defaultdict(list)
    all_metrics = defaultdict(list)
    
    with torch.inference_mode():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass (no teacher forcing for validation)
                predictions = model(batch, use_teacher_forcing=False)
                
                # Compute loss
                loss, loss_dict = loss_fn(predictions, batch)
                
                # Check for NaN/inf losses
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss in validation batch {batch_idx}")
                    continue
                
                # Record losses
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        epoch_losses[key].append(value.item())
                    else:
                        epoch_losses[key].append(value)
                
                # Compute metrics for the entire batch
                video_keys = []
                if 'location' in batch and 'video' in batch:
                    for i in range(len(batch['location'])):
                        loc = batch['location'][i] if isinstance(batch['location'], list) else batch['location'][i].item()
                        vid = batch['video'][i] if isinstance(batch['video'], list) else batch['video'][i].item() 
                        video_keys.append(f"{loc}_{vid}")
                
                # For batch metrics, we'll compute them without denormalization 
                # (or use a representative video_stats if all samples are from same video)
                video_stats = None
                if dataset and video_keys and all(k == video_keys[0] for k in video_keys):
                    # All samples from same video, use its stats
                    video_stats = dataset.get_video_stats(video_keys[0])
                
                # Compute metrics for the batch
                metrics = compute_metrics(predictions, batch, video_stats)
                for k, v in metrics.items():
                    if np.isfinite(v):  # Only record finite metrics
                        all_metrics[k].append(v)

                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'ADE': f"{metrics.get('ADE', 0.0):.3f}",
                    'Valid%': f"{loss_dict.get('valid_ratio', 0.0):.2f}"
                })
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                # Skip this batch and continue
                continue
    
    # Average losses and metrics
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}
    
    return avg_losses, avg_metrics

def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, 
                   val_metrics, save_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"New best model saved at epoch {epoch}")
    
    return checkpoint_path

def plot_training_curves(train_losses_history, val_losses_history, val_metrics_history, save_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot 1: Total Loss
    if 'total_loss' in train_losses_history and 'total_loss' in val_losses_history:
        axes[0].plot(train_losses_history['total_loss'], label='Train', alpha=0.7)
        axes[0].plot(val_losses_history['total_loss'], label='Validation', alpha=0.7)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: ADE
    if 'ADE' in val_metrics_history:
        axes[1].plot(val_metrics_history['ADE'], color='green', alpha=0.7)
        axes[1].set_title('Average Displacement Error (ADE)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ADE')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: FDE
    if 'FDE' in val_metrics_history:
        axes[2].plot(val_metrics_history['FDE'], color='red', alpha=0.7)
        axes[2].set_title('Final Displacement Error (FDE)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('FDE')
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Valid Position Ratio
    if 'Valid_Position_Ratio' in val_metrics_history:
        axes[3].plot(val_metrics_history['Valid_Position_Ratio'], color='purple', alpha=0.7)
        axes[3].set_title('Valid Position Ratio')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Ratio')
        axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Uncertainty (if available)
    if 'Avg_Uncertainty' in val_metrics_history:
        axes[4].plot(val_metrics_history['Avg_Uncertainty'], color='orange', alpha=0.7)
        axes[4].set_title('Average Uncertainty')
        axes[4].set_xlabel('Epoch')
        axes[4].set_ylabel('Uncertainty')
        axes[4].grid(True, alpha=0.3)
    
    # Plot 6: Position vs Delta Loss (if available)
    if 'position_loss' in val_losses_history and 'delta_loss' in val_losses_history:
        axes[5].plot(val_losses_history['position_loss'], label='Position', alpha=0.7)
        axes[5].plot(val_losses_history['delta_loss'], label='Delta', alpha=0.7)
        axes[5].set_title('Component Losses')
        axes[5].set_xlabel('Epoch')
        axes[5].set_ylabel('Loss')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function"""
    # Configuration
    config = {
        'drone_data_root': "../../square_stanford_data",
        'original_dataset_root': "../../stanford_data/archive",
        'classes': ['Pedestrian','Biker','Skater','Cart','Car','Bus'],
        
        # Model parameters
        'model_params': {
            'd_model': 64,
            'nhead': 8,
            'num_layers': 1,
            'T_past': 10,
            'T_future': 20,
            'use_deltas': True,
            'predict_uncertainty': True
        },
        
        # Training parameters
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'scheduler_type': 'cosine',  # 'cosine', 'step', 'plateau', or None
        
        # Loss parameters - Updated to handle multiple masks
        'loss_params': {
            'use_deltas': True,
            'predict_uncertainty': True,
            'position_weight': 1.0,
            'delta_weight': 1.0,
            'uncertainty_weight': 0.1,
            'use_existence_mask': True,
            'use_visibility_mask': True
        },
        
        # Data parameters
        'normalize_positions': False,
        'train_split': 0.1,
        'val_split': 0.9,
        'num_workers': 4,
        
        # Checkpointing
        'save_dir': './checkpoints_new',
        'save_every_epoch': True,
        'early_stopping_patience': 5
    }
    
    # Setup
    device = setup_device()
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create data loaders
    train_loader, val_loader, dataset = create_data_loaders(
        drone_data_root=config['drone_data_root'],
        original_dataroot=config['original_dataset_root'],
        classes=config['classes'],
        T_past=config['model_params']['T_past'],
        T_future=config['model_params']['T_future'],
        use_deltas=config['model_params']['use_deltas'],
        normalize_positions=config['normalize_positions'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        val_split=config['val_split'],
        num_workers=config['num_workers']
    )
    
    # Define locations for model initialization
    locations = [
        "bookstore",
        "coupa",
        "deathCircle",
        "gates",
        "hyang",
        "little",
        "nexus",
        "quad",
    ]
    
    # Initialize model
    model = initialize_model(
        num_classes=len(config['classes']),
        locations=locations,
        device=device,
        d_model=config['model_params']['d_model'],
        num_layers=config['model_params']['num_layers'],
        T_past=config['model_params']['T_past'],
        T_future=config['model_params']['T_future']
    )
    
    # Create loss function with updated parameters
    loss_fn = TrajectoryLoss(**config['loss_params'])
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler_type=config['scheduler_type'],
        num_epochs=config['num_epochs']
    )
    
    # Training history
    train_losses_history = defaultdict(list)
    val_losses_history = defaultdict(list)
    val_metrics_history = defaultdict(list)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print("-" * 50)
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            use_teacher_forcing=True
        )
        
        # Validate
        val_losses, val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            dataset=dataset
        )
        
        # Update learning rate
        if scheduler is not None:
            # Get the main loss for scheduling
            main_loss_key = 'total_loss' if 'total_loss' in val_losses else next(iter(val_losses.keys()), 'loss')
            if config['scheduler_type'] == 'plateau':
                scheduler.step(val_losses.get(main_loss_key, 0.0))
            else:
                scheduler.step()
        
        # Record history
        for key, value in train_losses.items():
            train_losses_history[key].append(value)
        
        for key, value in val_losses.items():
            val_losses_history[key].append(value)
        
        for key, value in val_metrics.items():
            val_metrics_history[key].append(value)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        main_loss_key = 'total_loss' if 'total_loss' in train_losses else next(iter(train_losses.keys()), 'loss')
        
        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        if main_loss_key in train_losses:
            print(f"Train Loss ({main_loss_key}): {train_losses[main_loss_key]:.4f}")
        if main_loss_key in val_losses:
            print(f"Val Loss ({main_loss_key}): {val_losses[main_loss_key]:.4f}")
        if 'ADE' in val_metrics:
            print(f"Val ADE: {val_metrics['ADE']:.4f}")
        if 'FDE' in val_metrics:
            print(f"Val FDE: {val_metrics['FDE']:.4f}")
        if 'Valid_Position_Ratio' in val_metrics:
            print(f"Valid Position Ratio: {val_metrics['Valid_Position_Ratio']:.3f}")
        if 'Avg_Uncertainty' in val_metrics:
            print(f"Avg Uncertainty: {val_metrics['Avg_Uncertainty']:.4f}")
        
        # Additional loss components
        if 'position_loss' in val_losses:
            print(f"  Position Loss: {val_losses['position_loss']:.4f}")
        if 'delta_loss' in val_losses:
            print(f"  Delta Loss: {val_losses['delta_loss']:.4f}")
        if 'uncertainty_reg' in val_losses:
            print(f"  Uncertainty Reg: {val_losses['uncertainty_reg']:.4f}")
        
        # Save checkpoint
        current_val_loss = val_losses.get(main_loss_key, float('inf'))
        is_best = current_val_loss < best_val_loss
        if is_best:
            best_val_loss = current_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if config['save_every_epoch'] or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                train_losses=train_losses_history,
                val_losses=val_losses_history,
                val_metrics=val_metrics_history,
                save_dir=config['save_dir'],
                is_best=is_best
            )
        
        # Plot training curves every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_training_curves(
                train_losses_history, val_losses_history, 
                val_metrics_history, config['save_dir']
            )
        
        # Early stopping
        if epochs_without_improvement >= config['early_stopping_patience']:
            print(f"\nEarly stopping after {config['early_stopping_patience']} epochs without improvement")
            break
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final training curves
    plot_training_curves(
        train_losses_history, val_losses_history, 
        val_metrics_history, config['save_dir']
    )
    
    # Save final results
    final_results = {
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
        'training_history': {
            'train_losses': train_losses_history,
            'val_losses': val_losses_history,
            'val_metrics': val_metrics_history
        }
    }
    
    with open(os.path.join(config['save_dir'], 'final_results.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        json.dump(convert_numpy(final_results), f, indent=2)

if __name__ == "__main__":
    main()


# def setup_device():
#     """Setup device for training"""
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         print(f"Using GPU: {torch.cuda.get_device_name()}")
#         print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
#     else:
#         device = torch.device('cpu')
#         print("Using CPU")
#     return device

# def create_data_loaders(drone_data_root, original_dataroot, classes, T_past=10, T_future=10, 
#                         use_deltas=True, normalize_positions=True, batch_size=32, 
#                         train_split=0.9, val_split=0.1, num_workers=4):
#     print('Loading dataset')
#     dataset = OptimizedAgentSequenceDataset(
#         drone_data_root, original_dataroot, classes, T_past, T_future, 
#         use_deltas=use_deltas, normalize_positions=normalize_positions
#     )
#     total_size = len(dataset)
#     train_size = int(train_split * total_size)
#     val_size = total_size - train_size
#     print("total size:", total_size)

#     train_dataset, val_dataset = torch.utils.data.random_split(
#         dataset, [train_size, val_size],
#         generator=torch.Generator().manual_seed(42)
#     )
    
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, 
#         num_workers=num_workers, pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False, 
#         num_workers=num_workers, pin_memory=True
#     )
#     return train_loader, val_loader, dataset

# def initialize_model(num_classes, locations, device, d_model=64, num_layers=2, T_past=10, T_future=10):
#     model = LandscapeAwareTrajectoryPredictor(
#         num_classes, locations, d_model=d_model, num_layers=num_layers, 
#         T_past=T_past, T_future=T_future
#     )
#     model = model.to(device)

#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"Model initialized:")
#     print(f"  Total parameters: {total_params:,}")
#     print(f"  Trainable parameters: {trainable_params:,}")
    
#     return model

# def create_optimizer_scheduler(model, learning_rate=1e-4, weight_decay=1e-5, 
#                              scheduler_type='cosine', num_epochs=100):
#     """Create optimizer and learning rate scheduler"""
#     optimizer = torch.optim.AdamW(
#         model.parameters(), 
#         lr=learning_rate, 
#         weight_decay=weight_decay
#     )
    
#     if scheduler_type == 'cosine':
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=num_epochs, eta_min=1e-6
#         )
#     elif scheduler_type == 'step':
#         scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer, step_size=30, gamma=0.5
#         )
#     elif scheduler_type == 'plateau':
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=10, verbose=True
#         )
#     else:
#         scheduler = None
    
#     return optimizer, scheduler

# def train_epoch(model, train_loader, optimizer, loss_fn, device, use_teacher_forcing=True):
#     model.train()
#     epoch_losses = defaultdict(list) 
#     pbar = tqdm(train_loader, desc="Training")
    
#     for batch_idx, batch in enumerate(pbar):
#         # Move batch to device
#         batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
#                 for k, v in batch.items()}
        
#         optimizer.zero_grad()

#         try:
#             predictions = model(batch, use_teacher_forcing=use_teacher_forcing)
#             loss, loss_dict = loss_fn(predictions, batch)
            
#             # Backward pass
#             loss.backward()
            
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()

#             # Record losses
#             for key, value in loss_dict.items():
#                 if isinstance(value, torch.Tensor):
#                     epoch_losses[key].append(value.item())
#                 else:
#                     epoch_losses[key].append(value)

#             pbar.set_postfix({
#                 'Loss': f"{loss.item():.4f}",
#                 'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
#             })
            
#         except Exception as e:
#             print(f"Error in training batch {batch_idx}: {e}")
#             # Skip this batch and continue
#             continue
    
#     avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
#     return avg_losses

# def validate_epoch(model, val_loader, loss_fn, device, dataset=None):
#     """Validate for one epoch"""
#     model.eval()
#     epoch_losses = defaultdict(list)
#     all_metrics = defaultdict(list)
    
#     with torch.inference_mode():
#         pbar = tqdm(val_loader, desc="Validation")
#         for batch_idx, batch in enumerate(pbar):
#             try:
#                 # Move batch to device
#                 batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
#                         for k, v in batch.items()}
                
#                 # Forward pass (no teacher forcing for validation)
#                 predictions = model(batch, use_teacher_forcing=False)
                
#                 # Compute loss
#                 loss, loss_dict = loss_fn(predictions, batch)
                
#                 # Record losses
#                 for key, value in loss_dict.items():
#                     if isinstance(value, torch.Tensor):
#                         epoch_losses[key].append(value.item())
#                     else:
#                         epoch_losses[key].append(value)
                
#                 # Compute metrics for each sample in batch
#                 batch_size = len(batch.get('location', batch.get('future_positions', [None])))
#                 for i in range(batch_size):
#                     # Extract single sample
#                     single_pred = {}
#                     single_target = {}
                    
#                     for key, value in predictions.items():
#                         if isinstance(value, torch.Tensor):
#                             single_pred[key] = value[i:i+1]
                    
#                     for key in ['future_positions', 'occ_mask', 'location', 'video']:
#                         if key in batch:
#                             if isinstance(batch[key], torch.Tensor):
#                                 single_target[key] = batch[key][i:i+1]
#                             elif isinstance(batch[key], list):
#                                 single_target[key] = [batch[key][i]]
#                             else:
#                                 single_target[key] = batch[key] if not hasattr(batch[key], '__getitem__') else batch[key][i]
                    
#                     # Get video stats for denormalization if available
#                     video_key = None
#                     if 'location' in single_target and 'video' in single_target:
#                         if isinstance(single_target['location'], list) and isinstance(single_target['video'], list):
#                             video_key = f"{single_target['location'][0]}_{single_target['video'][0]}"
                    
#                     video_stats = dataset.get_video_stats(video_key) if dataset and video_key else None
                    
#                     # Compute metrics
#                     metrics = compute_metrics(single_pred, single_target, video_stats)
#                     for k, v in metrics.items():
#                         all_metrics[k].append(v)
                
#                 pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
#             except Exception as e:
#                 print(f"Error in validation batch {batch_idx}: {e}")
#                 # Skip this batch and continue
#                 continue
    
#     # Average losses and metrics
#     avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
#     avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
#     return avg_losses, avg_metrics

# def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, 
#                    val_metrics, save_dir, is_best=False):
#     """Save model checkpoint"""
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'train_losses': train_losses,
#         'val_losses': val_losses,
#         'val_metrics': val_metrics
#     }
    
#     if scheduler is not None:
#         checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
#     # Save regular checkpoint
#     checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:03d}.pth')
#     torch.save(checkpoint, checkpoint_path)
    
#     # Save best model
#     if is_best:
#         best_path = os.path.join(save_dir, 'best_model.pth')
#         torch.save(checkpoint, best_path)
#         print(f"New best model saved at epoch {epoch}")
    
#     return checkpoint_path

# def plot_training_curves(train_losses_history, val_losses_history, val_metrics_history, save_dir):
#     """Plot and save training curves"""
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
#     # Get the first available loss key as the main loss
#     main_loss_key = next(iter(train_losses_history.keys())) if train_losses_history else 'loss'
    
#     # Loss curves
#     if main_loss_key in train_losses_history:
#         axes[0, 0].plot(train_losses_history[main_loss_key], label='Train')
#     if main_loss_key in val_losses_history:
#         axes[0, 0].plot(val_losses_history[main_loss_key], label='Validation')
#     axes[0, 0].set_title(f'{main_loss_key} Loss')
#     axes[0, 0].set_xlabel('Epoch')
#     axes[0, 0].set_ylabel('Loss')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True)
    
#     # ADE metric
#     if 'ADE' in val_metrics_history:
#         axes[0, 1].plot(val_metrics_history['ADE'])
#         axes[0, 1].set_title('Average Displacement Error (ADE)')
#         axes[0, 1].set_xlabel('Epoch')
#         axes[0, 1].set_ylabel('ADE')
#         axes[0, 1].grid(True)
    
#     # FDE metric
#     if 'FDE' in val_metrics_history:
#         axes[1, 0].plot(val_metrics_history['FDE'])
#         axes[1, 0].set_title('Final Displacement Error (FDE)')
#         axes[1, 0].set_xlabel('Epoch')
#         axes[1, 0].set_ylabel('FDE')
#         axes[1, 0].grid(True)
    
#     # Uncertainty metrics (if available)
#     if 'Avg_Uncertainty' in val_metrics_history:
#         axes[1, 1].plot(val_metrics_history['Avg_Uncertainty'])
#         axes[1, 1].set_title('Average Uncertainty')
#         axes[1, 1].set_xlabel('Epoch')
#         axes[1, 1].set_ylabel('Uncertainty')
#         axes[1, 1].grid(True)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def main():
#     """Main training function"""
#     # Configuration
#     config = {
#         'drone_data_root': "../../square_stanford_data",
#         'original_dataset_root': "../../stanford_data/archive",
#         'classes': ['Pedestrian','Biker','Skater','Cart','Car','Bus'],
        
#         # Model parameters
#         'model_params': {
#             'd_model': 64,
#             'nhead': 8,
#             'num_layers': 1,
#             'T_past': 10,
#             'T_future': 20,
#             'use_deltas': True,
#             'predict_uncertainty': True
#         },
        
#         # Training parameters
#         'batch_size': 32,
#         'num_epochs': 10,
#         'learning_rate': 1e-4,
#         'weight_decay': 1e-5,
#         'scheduler_type': 'cosine',  # 'cosine', 'step', 'plateau', or None
        
#         # Loss parameters
#         'loss_params': {
#             'use_deltas': True,
#             'predict_uncertainty': True,
#             'position_weight': 1.0,
#             'delta_weight': 1.0,
#             'uncertainty_weight': 0.1
#         },
        
#         # Data parameters
#         'normalize_positions': False,
#         'train_split': 0.1,
#         'val_split': 0.9,
#         'num_workers': 4,
        
#         # Checkpointing
#         'save_dir': './checkpoints_new',
#         'save_every_epoch': True,
#         'early_stopping_patience': 5
#     }
    
#     # Setup
#     device = setup_device()
#     os.makedirs(config['save_dir'], exist_ok=True)
    
#     # Save configuration
#     with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
#         json.dump(config, f, indent=2)
    
#     # Create data loaders
#     train_loader, val_loader, dataset = create_data_loaders(
#         drone_data_root=config['drone_data_root'],
#         original_dataroot=config['original_dataset_root'],
#         classes=config['classes'],
#         T_past=config['model_params']['T_past'],
#         T_future=config['model_params']['T_future'],
#         use_deltas=config['model_params']['use_deltas'],
#         normalize_positions=config['normalize_positions'],
#         batch_size=config['batch_size'],
#         train_split=config['train_split'],
#         val_split=config['val_split'],
#         num_workers=config['num_workers']
#     )
    
#     # Define locations for model initialization
#     locations = [
#         "bookstore",
#         "coupa",
#         "deathCircle",
#         "gates",
#         "hyang",
#         "little",
#         "nexus",
#         "quad",
#     ]
    
#     # Initialize model
#     model = initialize_model(
#         num_classes=len(config['classes']),
#         locations=locations,
#         device=device,
#         d_model=config['model_params']['d_model'],
#         num_layers=config['model_params']['num_layers'],
#         T_past=config['model_params']['T_past'],
#         T_future=config['model_params']['T_future']
#     )
    
#     # Create loss function
#     loss_fn = TrajectoryLoss(**config['loss_params'])
    
#     # Create optimizer and scheduler
#     optimizer, scheduler = create_optimizer_scheduler(
#         model=model,
#         learning_rate=config['learning_rate'],
#         weight_decay=config['weight_decay'],
#         scheduler_type=config['scheduler_type'],
#         num_epochs=config['num_epochs']
#     )
    
#     # Training history
#     train_losses_history = defaultdict(list)
#     val_losses_history = defaultdict(list)
#     val_metrics_history = defaultdict(list)
    
#     best_val_loss = float('inf')
#     epochs_without_improvement = 0
    
#     print(f"\nStarting training for {config['num_epochs']} epochs...")
#     print(f"Device: {device}")
#     print(f"Batch size: {config['batch_size']}")
#     print(f"Learning rate: {config['learning_rate']}")
#     print("-" * 50)
    
#     for epoch in range(config['num_epochs']):
#         start_time = time.time()
        
#         print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
#         # Train
#         train_losses = train_epoch(
#             model=model,
#             train_loader=train_loader,
#             loss_fn=loss_fn,
#             optimizer=optimizer,
#             device=device,
#             use_teacher_forcing=True
#         )
        
#         # Validate
#         val_losses, val_metrics = validate_epoch(
#             model=model,
#             val_loader=val_loader,
#             loss_fn=loss_fn,
#             device=device,
#             dataset=dataset
#         )
        
#         # Update learning rate
#         if scheduler is not None:
#             # Get the main loss for scheduling
#             main_loss_key = next(iter(val_losses.keys())) if val_losses else 'loss'
#             if config['scheduler_type'] == 'plateau':
#                 scheduler.step(val_losses.get(main_loss_key, 0.0))
#             else:
#                 scheduler.step()
        
#         # Record history
#         for key, value in train_losses.items():
#             train_losses_history[key].append(value)
        
#         for key, value in val_losses.items():
#             val_losses_history[key].append(value)
        
#         for key, value in val_metrics.items():
#             val_metrics_history[key].append(value)
        
#         # Print epoch results
#         epoch_time = time.time() - start_time
#         main_loss_key = next(iter(train_losses.keys())) if train_losses else 'loss'
        
#         print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
#         if main_loss_key in train_losses:
#             print(f"Train Loss ({main_loss_key}): {train_losses[main_loss_key]:.4f}")
#         if main_loss_key in val_losses:
#             print(f"Val Loss ({main_loss_key}): {val_losses[main_loss_key]:.4f}")
#         if 'ADE' in val_metrics:
#             print(f"Val ADE: {val_metrics['ADE']:.4f}")
#         if 'FDE' in val_metrics:
#             print(f"Val FDE: {val_metrics['FDE']:.4f}")
        
#         # Save checkpoint
#         current_val_loss = val_losses.get(main_loss_key, float('inf'))
#         is_best = current_val_loss < best_val_loss
#         if is_best:
#             best_val_loss = current_val_loss
#             epochs_without_improvement = 0
#         else:
#             epochs_without_improvement += 1
        
#         if config['save_every_epoch'] or is_best:
#             save_checkpoint(
#                 model=model,
#                 optimizer=optimizer,
#                 scheduler=scheduler,
#                 epoch=epoch + 1,
#                 train_losses=train_losses_history,
#                 val_losses=val_losses_history,
#                 val_metrics=val_metrics_history,
#                 save_dir=config['save_dir'],
#                 is_best=is_best
#             )
        
#         # Plot training curves every 5 epochs
#        # if (epoch + 1) % 5 == 0:
#         #    plot_training_curves(
#          #       train_losses_history, val_losses_history, 
#           #      val_metrics_history, config['save_dir']
#            # )
        
#         # Early stopping
#         if epochs_without_improvement >= config['early_stopping_patience']:
#             print(f"\nEarly stopping after {config['early_stopping_patience']} epochs without improvement")
#             break
    
#     print("\nTraining completed!")
#     print(f"Best validation loss: {best_val_loss:.4f}")
    
#     # Save final results
#     final_results = {
#         'best_val_loss': best_val_loss,
#         'training_history': {
#             'train_losses': train_losses_history,
#             'val_losses': val_losses_history,
#             'val_metrics': val_metrics_history
#         }
#     }
    
#     with open(os.path.join(config['save_dir'], 'final_results.json'), 'w') as f:
#         # Convert numpy types to Python types for JSON serialization
#         def convert_numpy(obj):
#             if isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             elif isinstance(obj, np.generic):
#                 return obj.item()
#             elif isinstance(obj, dict):
#                 return {k: convert_numpy(v) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [convert_numpy(v) for v in obj]
#             return obj
        
#         json.dump(convert_numpy(final_results), f, indent=2)

# if __name__ == "__main__":
#     main()
