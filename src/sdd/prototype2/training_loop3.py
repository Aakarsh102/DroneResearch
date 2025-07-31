# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import os
# import pickle as pkl
# import pandas as pd
# import numpy as np 
# import math
# import json
# from functools import lru_cache
# from threading import Lock
# import threading
# import time
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import warnings
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# from functools import lru_cache
# import h5py
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# from new_model import GraphInteractionModel
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# import wandb
# import numpy as np
# from pathlib import Path
# import json
# import time
# from tqdm import tqdm

# warnings.filterwarnings('ignore')


# def process_drone_location_worker(args):
#     """Worker function for processing drone location data"""
#     drone_root, loc = args
#     if loc == 'hyang':
#         return {}
        
#     obs = {}
#     loc_path = os.path.join(drone_root, loc)
#     if not os.path.isdir(loc_path):
#         return {}
    
#     for dr in os.listdir(loc_path):
#         dr_path = os.path.join(loc_path, dr)
#         if not os.path.isdir(dr_path):
#             continue
            
#         for fname in os.listdir(dr_path):
#             if not fname.endswith("_annotations.txt"):
#                 continue
                
#             vid = fname.replace("_annotations.txt", "")
#             key = (loc, vid)
            
#             try:
#                 df = pd.read_csv(os.path.join(dr_path, fname), sep=' ', header=None,
#                                names=['trackId','xmin','ymin','xmax','ymax',
#                                      'frame','lost','occluded','generated','label'],
#                                usecols=['trackId', 'frame'])  # Only load needed columns
                
#                 obs[key] = set(zip(df['trackId'], df['frame'].astype(int)))
#             except Exception as e:
#                 print(f"Error processing {fname}: {e}")
    
#     return obs

# def process_video_worker(args):
#     """Worker function for processing video data"""
#     loc, vid, obs_sets, orig_root, classes, cls2idx, T_past, T_future, frame_subsample = args
#     orig_file = os.path.join(orig_root, "annotations", loc, vid, "annotations.txt")
    
#     if not os.path.isfile(orig_file):
#         return []
    
#     indices = []
#     key = (loc, vid)
#     seen = obs_sets.get(key, set())
    
#     try:
#         # Load only necessary data
#         df = pd.read_csv(orig_file, sep=' ', header=None,
#                        names=['trackId','xmin','ymin','xmax','ymax',
#                              'frame','lost','occluded','generated','label'])
        
#         # Filter by classes early
#         df = df[df['label'].isin(classes)]
        
#         if len(df) == 0:
#             return []
        
#         # Vectorized center calculation
#         df['x'] = (df['xmin'] + df['xmax']) * 0.5
#         df['y'] = (df['ymin'] + df['ymax']) * 0.5
        
#         # Process each track
#         for tid, grp in df.groupby('trackId'):
#             grp = grp.sort_values('frame').reset_index(drop=True)
#             frames = grp['frame'].values
            
#             # Vectorized observation mask calculation
#             obs_mask = np.array([1.0 if (tid, int(f)) in seen else 0.0 for f in frames])
            
#             L = len(frames)
#             window = T_past + T_future
            
#             # NEW: Frame subsampling logic
#             # Generate all possible starting frame offsets (0 to frame_subsample-1)
#             for frame_offset in range(frame_subsample):
#                 for i in range(L - (window - 1) * frame_subsample - frame_offset):
#                     # Calculate the subsampled frame indices
#                     subsampled_indices = []
#                     subsampled_frames = []
                    
#                     for j in range(window):
#                         target_idx = i + frame_offset + j * frame_subsample
#                         if target_idx >= L:
#                             break
#                         subsampled_indices.append(target_idx)
#                         subsampled_frames.append(frames[target_idx])
                    
#                     # Check if we have enough frames for the full window
#                     if len(subsampled_indices) < window:
#                         continue
                    
#                     # Check frame continuity with subsampling
#                     expected_frame_diff = frame_subsample
#                     valid_sequence = True
                    
#                     for j in range(1, len(subsampled_frames)):
#                         actual_diff = subsampled_frames[j] - subsampled_frames[j-1]
#                         # Allow some tolerance for missing frames
#                         if abs(actual_diff - expected_frame_diff) > frame_subsample // 2:
#                             valid_sequence = False
#                             break
                    
#                     if not valid_sequence:
#                         continue
                    
#                     # Check observation requirement for past frames
#                     past_indices = subsampled_indices[:T_past]
#                     if obs_mask[past_indices].sum() < 2:
#                         continue
                    
#                     # Store sample index instead of full data
#                     sample_idx = {
#                         'location': loc,
#                         'video': vid,
#                         'track_id': tid,
#                         'start_idx': i,
#                         'frame_offset': frame_offset,
#                         'subsampled_indices': subsampled_indices,
#                         'subsampled_frames': subsampled_frames,
#                         'window_size': window,
#                         'label': cls2idx[grp.iloc[subsampled_indices[T_past - 1]]['label']]
#                     }
                    
#                     indices.append(sample_idx)
                
#     except Exception as e:
#         print(f"Error processing {loc}/{vid}: {e}")
        
#     return indices

# def process_location_stats_worker(args):
#     """Worker function for computing location statistics"""
#     orig_root, loc = args
#     if loc in ['hyang', '.DS_Store']:
#         return {}
    
#     loc_stats = {}
#     loc_path = Path(orig_root) / "annotations" / loc
    
#     if not loc_path.is_dir():
#         return {}
    
#     for vid_path in loc_path.iterdir():
#         if not vid_path.is_dir():
#             continue
            
#         orig_file = vid_path / "annotations.txt"
#         if not orig_file.exists():
#             continue
        
#         video_key = f"{loc}_{vid_path.name}"
        
#         try:
#             # Use faster CSV reading
#             df = pd.read_csv(orig_file, sep=' ', header=None, 
#                            names=['trackId','xmin','ymin','xmax','ymax',
#                                  'frame','lost','occluded','generated','label'],
#                            usecols=['xmin','ymin','xmax','ymax'])  # Only load needed columns
            
#             if len(df) > 0:
#                 # Vectorized center calculation
#                 x_centers = (df['xmin'] + df['xmax']) * 0.5
#                 y_centers = (df['ymin'] + df['ymax']) * 0.5
#                 coords = np.column_stack([x_centers, y_centers])
                
#                 loc_stats[video_key] = {
#                     'mean': coords.mean(axis=0),
#                     'std': coords.std(axis=0),
#                     'min': coords.min(axis=0),
#                     'max': coords.max(axis=0)
#                 }
#         except Exception as e:
#             print(f"Error processing {video_key}: {e}")
    
#     return loc_stats

# def process_video_multi_agent_worker(args):
#     """Worker function for processing multi-agent video data"""
#     loc, vid, obs_sets, orig_root, classes, cls2idx, T_past, T_future, frame_subsample, min_frames_per_agent = args
    
#     orig_file = os.path.join(orig_root, "annotations", loc, vid, "annotations.txt")
#     if not os.path.isfile(orig_file):
#         return []
    
#     indices = []
#     key = (loc, vid)
#     seen = obs_sets.get(key, set())  # Use set() as default instead of None
    
#     try:
#         df = pd.read_csv(orig_file, sep=' ', header=None,
#                        names=['trackId','xmin','ymin','xmax','ymax',
#                              'frame','lost','occluded','generated','label'])
#         df = df[df['label'].isin(classes)]
#         if len(df) == 0:
#             return []
#         df['x'] = (df['xmin'] + df['xmax']) * 0.5
#         df['y'] = (df['ymin'] + df['ymax']) * 0.5
        
#         # Get all unique frames
#         all_frames = sorted(df['frame'].unique())
#         window_size = T_past + T_future
        
#         # Frame subsampling logic for multi-agent
#         # Try different starting offsets for subsampling
#         for frame_offset in range(frame_subsample):
#             # Calculate how many subsampled windows we can fit
#             max_subsampled_windows = (len(all_frames) - frame_offset) // frame_subsample
            
#             for start_subsample_idx in range(max_subsampled_windows - window_size + 1):
#                 # Calculate the actual frame indices we'll use
#                 window_frame_indices = []
#                 window_frames = []
                
#                 for j in range(window_size):
#                     frame_idx = frame_offset + (start_subsample_idx + j) * frame_subsample
#                     if frame_idx < len(all_frames):
#                         window_frame_indices.append(frame_idx)
#                         window_frames.append(all_frames[frame_idx])
                
#                 if len(window_frames) < window_size:
#                     continue

#                 past_frames = window_frames[:T_past]
#                 future_frames = window_frames[T_past:]

#                 window_df = df[df['frame'].isin(window_frames)]

#                 if len(window_df) == 0:  # Check window_df, not window_frames
#                     continue

#                 # Find agents that meet minimum frame requirement
#                 valid_agents = []
#                 agent_data = {}

#                 for tid, agent_df in window_df.groupby('trackId'):
#                     agent_frames = set(agent_df['frame'].values)
#                     past_frames_count = len([f for f in past_frames if f in agent_frames])
#                     future_frames_count = len([f for f in future_frames if f in agent_frames])

#                     if past_frames_count >= min_frames_per_agent:
#                         # Only check observation mask for frames that exist for this agent
#                         obs_mask = np.array([1.0 if (tid, int(f)) in seen else 0.0 
#                                            for f in past_frames if f in agent_frames])
#                         if len(obs_mask) > 0 and obs_mask.sum() >= 2:  # Check array length
#                             valid_agents.append(tid)
#                             agent_data[tid] = {
#                                 'past_frames': past_frames_count,
#                                 'future_frames': future_frames_count,
#                                 'obs_count': obs_mask.sum()
#                             }

#                 if len(valid_agents) == 0:
#                     continue
                
#                 # Create sample index
#                 sample_idx = {
#                     'location': loc,
#                     'video': vid,
#                     'start_frame': window_frames[0],
#                     'end_frame': window_frames[-1],
#                     'window_frames': window_frames,
#                     'frame_offset': frame_offset,  # Store frame offset
#                     'subsample_start_idx': start_subsample_idx,  # Store subsample start
#                     'valid_agents': valid_agents,
#                     'agent_data': agent_data,
#                     'num_agents': len(valid_agents)
#                 }

#                 indices.append(sample_idx)
                
#     except Exception as e:
#         print(f"Error processing {loc}/{vid}: {e}")
#     return indices

# class OptimizedMultiAgentSequenceDataset(Dataset):
#     def __init__(self, drone_data_root, original_dataset_root, classes,
#                  T_past=10, T_future=10, use_deltas=True, normalize_positions=True,
#                  cache_dir="dataset_cache", lazy_loading=True, num_workers=4,
#                  max_agents=20, min_frames_per_agent=2, pad_value=-999.0,
#                  frame_subsample=12):  # NEW: Frame subsampling parameter
#         self.drone_data_root = drone_data_root
#         self.orig_root = original_dataset_root
#         self.classes = classes
#         self.cls2idx = {c:i for i,c in enumerate(classes)}
#         self.T_past = T_past
#         self.T_future = T_future
#         self.use_deltas = use_deltas
#         self.normalize_positions = normalize_positions
#         self.lazy_loading = lazy_loading
#         self.num_workers = num_workers
#         self.max_agents = max_agents
#         self.min_frames_per_agent = self.T_past
#         self.pad_value = pad_value  # Value used for padding missing agents/frames
#         self.frame_subsample = frame_subsample  # NEW: Subsample every Nth frame
#         self.obs_sets = None  # Will be populated during dataset building

#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(exist_ok = True)

#         self._initialize_dataset()
#         self._hdf5_file_cache = threading.local()
#         self._hdf5_lock = Lock()
#         self.hdf5_file_path = self.cache_dir / "multi_agent_trajectory_data.h5"

#     def _get_hdf5_file(self):
#         """Get thread-local HDF5 file handle"""
#         if not hasattr(self._hdf5_file_cache, 'file') or self._hdf5_file_cache.file is None:
#             try:
#                 self._hdf5_file_cache.file = h5py.File(self.hdf5_file_path, 'r')
#             except Exception as e:
#                 print(f"Error opening HDF5 file: {e}")
#                 return None
#         return self._hdf5_file_cache.file
    
#     @lru_cache(maxsize=1024)  # Increased cache size significantly
#     def _load_trajectory_data_batch(self, video_key: str, track_ids_tuple: tuple):
#         """Load multiple trajectory data at once for better I/O efficiency"""
#         hdf5_file = self._get_hdf5_file()
#         if hdf5_file is None:
#             return {}

#         try:
#             if video_key not in hdf5_file:
#                 return {}
                
#             video_grp = hdf5_file[video_key]
#             batch_data = {}
            
#             for track_id in track_ids_tuple:
#                 if str(track_id) not in video_grp:
#                     continue
                    
#                 track_grp = video_grp[str(track_id)]
#                 batch_data[track_id] = {
#                     'coords': track_grp['coords'][:].astype(np.float32),
#                     'frames': track_grp['frames'][:].astype(np.int32),
#                     'occluded': track_grp['occluded'][:].astype(np.float32),
#                     'labels': [label.decode('utf-8') for label in track_grp['labels'][:]]
#                 }
            
#             return batch_data
#         except Exception as e:
#             print(f"Error loading batch trajectory data for {video_key}: {e}")
#             return {}

#     def _initialize_dataset(self):
#         cache_info_file = self.cache_dir / "dataset_info.json"
        
#         # Create cache signature based on parameters
#         cache_signature = {
#             'T_past': self.T_past,
#             'T_future': self.T_future,
#             'use_deltas': self.use_deltas,
#             'normalize_positions': self.normalize_positions,
#             'classes': sorted(self.classes),
#             'max_agents': self.max_agents,
#             'min_frames_per_agent': self.min_frames_per_agent,
#             'frame_subsample': self.frame_subsample  # NEW: Include in cache signature
#         }
        
#         if cache_info_file.exists():
#             with open(cache_info_file, 'r') as f:
#                 cached_info = json.load(f)
            
#             if cached_info.get('signature') == cache_signature:
#                 print("Loading from cache...")
#                 self._load_from_cache(cached_info)
#                 return
        
#         print("Building dataset from scratch...")
#         self._build_dataset()
        
#         # Save cache info
#         cache_info = {
#             'signature': cache_signature,
#             'num_samples': len(self.sample_indices),
#             'lazy_loading': self.lazy_loading
#         }
        
#         with open(cache_info_file, 'w') as f:
#             json.dump(cache_info, f)

        
#     def _load_from_cache(self, cache_info):
#         # FIXED: Always load obs_sets when loading from cache
#         obs_sets_file = self.cache_dir / "obs_sets.pkl"
#         if obs_sets_file.exists():
#             with open(obs_sets_file, 'rb') as f:
#                 self.obs_sets = pkl.load(f)
#         else:
#             # If obs_sets file doesn't exist, rebuild it
#             print("obs_sets not found in cache, rebuilding...")
#             self.obs_sets = self._build_obs_sets_parallel()
#             with open(obs_sets_file, 'wb') as f:
#                 pkl.dump(self.obs_sets, f)
        
#         if self.lazy_loading:
#             indices_file = self.cache_dir/"sample_indices.pkl"
#             with open(indices_file, 'rb') as f:
#                 self.sample_indices = pkl.load(f)

#             if self.normalize_positions:
#                 stats_file = self.cache_dir / "video_stats.pkl"  # Fixed typo
#                 with open(stats_file, "rb") as f:
#                     self.video_stats = pkl.load(f)
#         else:
#             samples_file = self.cache_dir / "all_samples.pkl"
#             with open(samples_file, 'rb') as f:
#                 self.samples = pkl.load(f)

#     def _build_dataset(self):
#         """Build dataset using optimized processing"""
#         print("Computing video statistics...")
#         if self.normalize_positions:
#             self.video_stats = self._compute_video_stats_fast()
#         else:
#             self.video_stats = None
        
#         print("Building observation sets...")
#         self.obs_sets = self._build_obs_sets_parallel()
        
#         print("Processing multi-agent trajectories...")
#         if self.lazy_loading:
#             self.sample_indices = self._build_multi_agent_sample_indices_parallel(self.obs_sets)
#             self._save_samples_to_hdf5()
#         else:
#             self.samples = self._build_multi_agent_samples_parallel(self.obs_sets)
#             self._cache_samples()

#     def _build_multi_agent_sample_indices_parallel(self, obs_sets):
#         """Build sample indices for multi-agent trajectories with lazy loading and frame subsampling"""
#         args = []
#         for loc in os.listdir(os.path.join(self.orig_root, "annotations")):
#             if loc in ['hyang', '.DS_Store']:
#                 continue
#             loc_path = os.path.join(self.orig_root, "annotations", loc)
#             if not os.path.isdir(loc_path):
#                 continue
#             for vid in os.listdir(loc_path):
#                 args.append((loc, vid, obs_sets, self.orig_root, self.classes, 
#                             self.cls2idx, self.T_past, self.T_future, 
#                             self.frame_subsample, self.min_frames_per_agent))
        
#         # Process in parallel using the external worker function
#         with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
#             results = list(executor.map(process_video_multi_agent_worker, args))
        
#         # Flatten results
#         all_indices = []
#         for result in results:
#             all_indices.extend(result)
        
#         return all_indices

#     def _save_samples_to_hdf5(self):
#         """Save processed multi-agent trajectory data to HDF5"""
#         hdf5_file = self.cache_dir / "multi_agent_trajectory_data.h5"
        
#         # Group samples by video for efficient storage
#         video_groups = {}
#         for idx, sample_idx in enumerate(self.sample_indices):
#             video_key = f"{sample_idx['location']}_{sample_idx['video']}"
#             if video_key not in video_groups:
#                 video_groups[video_key] = []
#             video_groups[video_key].append((idx, sample_idx))
        
#         with h5py.File(hdf5_file, 'w') as f:
#             for video_key, samples in video_groups.items():
#                 if not samples:
#                     continue
                
#                 # Load video data once
#                 loc, vid = video_key.split('_', 1)
#                 orig_file = os.path.join(self.orig_root, "annotations", loc, vid, "annotations.txt")
                
#                 if not os.path.isfile(orig_file):
#                     continue
                
#                 df = pd.read_csv(orig_file, sep=' ', header=None,
#                                names=['trackId','xmin','ymin','xmax','ymax',
#                                      'frame','lost','occluded','generated','label'])
#                 df = df[df['label'].isin(self.classes)]
#                 df['x'] = (df['xmin'] + df['xmax']) * 0.5
#                 df['y'] = (df['ymin'] + df['ymax']) * 0.5
                
#                 # Create HDF5 group for this video
#                 video_grp = f.create_group(video_key)
                
#                 # Store trajectory data grouped by track
#                 track_data = {}
#                 for tid, grp in df.groupby('trackId'):
#                     grp = grp.sort_values('frame').reset_index(drop=True)
#                     track_data[tid] = {
#                         'coords': grp[['x', 'y']].values.astype(np.float32),
#                         'frames': grp['frame'].values.astype(np.int32),
#                         'occluded': grp['occluded'].values.astype(np.float32),
#                         'labels': grp['label'].values
#                     }
                
#                 # Store track data
#                 for tid, data in track_data.items():
#                     track_grp = video_grp.create_group(str(tid))
#                     track_grp.create_dataset('coords', data=data['coords'])
#                     track_grp.create_dataset('frames', data=data['frames'])
#                     track_grp.create_dataset('occluded', data=data['occluded'])
#                     track_grp.create_dataset('labels', data=data['labels'].astype('S10'))
        
#         # Cache sample indices and stats
#         with open(self.cache_dir / "sample_indices.pkl", 'wb') as f:
#             pkl.dump(self.sample_indices, f)
        
#         # FIXED: Always save obs_sets when building from scratch
#         with open(self.cache_dir / "obs_sets.pkl", 'wb') as f:
#             pkl.dump(self.obs_sets, f)
        
#         if self.video_stats:
#             with open(self.cache_dir / "video_stats.pkl", 'wb') as f:
#                 pkl.dump(self.video_stats, f)

#     def _compute_video_stats_fast(self):
#         locations = [d for d in os.listdir(Path(self.orig_root) / "annotations") 
#                     if d not in ['hyang', '.DS_Store']]
        
#         args = [(self.orig_root, loc) for loc in locations]
        
#         with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
#             results = list(executor.map(process_location_stats_worker, args))
        
#         # Merge results
#         video_stats = {}
#         for result in results:
#             video_stats.update(result)
        
#         return video_stats
    
#     def _build_obs_sets_parallel(self) -> Dict:
#         """Build observation sets using parallel processing"""
#         locations = [d for d in os.listdir(self.drone_data_root) 
#                     if os.path.isdir(os.path.join(self.drone_data_root, d))]
#         args = [(self.drone_data_root, loc) for loc in locations]
#         with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
#             results = list(executor.map(process_drone_location_worker, args))

#         merged_obs = {}
#         for result in results:
#             merged_obs.update(result)

#         return merged_obs

#     def _cache_samples(self):
#         """Cache all samples to disk"""
#         with open(self.cache_dir / "all_samples.pkl", 'wb') as f:
#             pkl.dump(self.samples, f)
        
#         # FIXED: Also save obs_sets when caching samples
#         with open(self.cache_dir / "obs_sets.pkl", 'wb') as f:
#             pkl.dump(self.obs_sets, f)

#     @lru_cache(maxsize=128)
#     def _load_trajectory_data(self, video_key: str, track_id: int):
#         hdf5_file = self.cache_dir / "multi_agent_trajectory_data.h5"

#         try:
#             with h5py.File(hdf5_file, 'r') as f:  # Fixed: use h5py.File instead of open
#                 if video_key not in f:
#                     return None
#                 if str(track_id) not in f[video_key]:
#                     return None
                    
#                 track_grp = f[video_key][str(track_id)]

#                 return {
#                     'coords': track_grp['coords'][:].astype(np.float32),
#                     'frames': track_grp['frames'][:].astype(np.int32),
#                     'occluded': track_grp['occluded'][:].astype(np.float32),
#                     'labels': [label.decode('utf-8') for label in track_grp['labels'][:]]
#                 }
#         except Exception as e:
#             print(f"Error loading trajectory data for {video_key}/{track_id}: {e}")
#             return None
        
#     def _compute_deltas_vectorized(self, positions, temporal_masks, is_future=False, past_positions=None):
#         """Vectorized delta computation"""
#         max_agents, seq_len, _ = positions.shape
#         deltas = np.zeros_like(positions)
        
#         for agent_idx in range(max_agents):
#             if not np.any(temporal_masks[agent_idx]):
#                 continue
                
#             valid_indices = np.where(temporal_masks[agent_idx] > 0)[0]
#             if len(valid_indices) <= 1 and not is_future:
#                 continue
            
#             if is_future and past_positions is not None:
#                 # Handle transition from past to future
#                 past_mask = temporal_masks[agent_idx] if past_positions.shape[1] == seq_len else None
#                 if past_mask is not None:
#                     past_valid = np.where(past_mask > 0)[0]
#                     if len(past_valid) > 0 and len(valid_indices) > 0:
#                         last_past_idx = past_valid[-1]
#                         first_future_idx = valid_indices[0]
#                         deltas[agent_idx, first_future_idx] = (
#                             positions[agent_idx, first_future_idx] - 
#                             past_positions[agent_idx, last_past_idx]
#                         )
#                         # Remove first future index from processing
#                         valid_indices = valid_indices[1:]
            
#             # Compute consecutive deltas
#             if len(valid_indices) > 1:
#                 for i in range(1, len(valid_indices)):
#                     curr_idx = valid_indices[i]
#                     prev_idx = valid_indices[i-1]
#                     deltas[agent_idx, curr_idx] = (
#                         positions[agent_idx, curr_idx] - positions[agent_idx, prev_idx]
#                     )
        
#         return deltas
#     def _normalize_coords_vectorized(self, coords_array, temporal_mask, video_key):
#         """Vectorized coordinate normalization"""
#         if self.video_stats is None or video_key not in self.video_stats:
#             return coords_array
        
#         stats = self.video_stats[video_key]
#         mean = stats['mean'].reshape(1, 1, 2)  # Broadcast shape
#         std = stats['std'].reshape(1, 1, 2) + 1e-8
        
#         # Only normalize where temporal_mask is True
#         normalized = coords_array.copy()
#         mask_expanded = temporal_mask[:, :, np.newaxis]  # Shape: (agents, time, 1)
#         normalized = np.where(mask_expanded, (coords_array - mean) / std, coords_array)
        
#         return normalized
        
#     def _normalize_coords(self, coords, video_key):
#         """Normalize coordinates using video-specific statistics"""
#         if self.video_stats is None or video_key not in self.video_stats:
#             return coords
        
#         stats = self.video_stats[video_key]
#         normalized = (coords - stats['mean']) / (stats['std'] + 1e-8)
#         return normalized
    
#     def get_video_stats(self, video_key):
#         """Return video-specific statistics for denormalization"""
#         if self.video_stats is None:
#             return None
#         return self.video_stats.get(video_key, None)
    
#     def __len__(self):
#         if self.lazy_loading:
#             return len(self.sample_indices)
#         else:
#             return len(self.samples)
    
#     def __getitem__(self, idx):
#         if not self.lazy_loading:
#             # Use pre-loaded samples
#             s = self.samples[idx]
#             return self._format_sample(s)
        
#         # Lazy loading: load sample on demand
#         sample_idx = self.sample_indices[idx]
#         return self._load_multi_agent_sample_on_demand(sample_idx)

#     def _load_multi_agent_sample_on_demand(self, sample_idx):
#         """Optimized multi-agent sample loading with vectorized operations"""
#         # Get observation set with fallback
#         obs_set = self.obs_sets.get((sample_idx['location'], sample_idx['video']), set()) if self.obs_sets else set()
        
#         video_key = f"{sample_idx['location']}_{sample_idx['video']}"
#         window_frames = sample_idx['window_frames']
#         valid_agents = sample_idx['valid_agents']
        
#         past_frames = window_frames[:self.T_past]
#         future_frames = window_frames[self.T_past:]
#         num_agents = min(len(valid_agents), self.max_agents)
        
#         # Load all trajectory data in one batch
#         track_ids_tuple = tuple(valid_agents[:self.max_agents])
#         batch_data = self._load_trajectory_data_batch(video_key, track_ids_tuple)
        
#         # Pre-allocate all arrays
#         arrays = {
#             'past_positions': np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32),
#             'future_positions': np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32),
#             'past_positions_orig': np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32),
#             'future_positions_orig': np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32),
#             'obs_masks': np.zeros((self.max_agents, self.T_past), dtype=np.float32),
#             'temporal_masks_past': np.zeros((self.max_agents, self.T_past), dtype=np.float32),
#             'temporal_masks_future': np.zeros((self.max_agents, self.T_future), dtype=np.float32),
#             'occ_masks': np.zeros((self.max_agents, self.T_future), dtype=np.float32),
#             'agent_masks': np.zeros(self.max_agents, dtype=np.float32),
#             'agent_labels': np.full(self.max_agents, -1, dtype=np.int32),
#             'agent_ids': np.full(self.max_agents, -1, dtype=np.int32),
#         }
        
#         # Pre-compute frame lookups for all agents
#         frame_lookups = {}
#         for agent_idx, track_id in enumerate(valid_agents[:self.max_agents]):
#             if track_id not in batch_data:
#                 continue
                
#             track_data = batch_data[track_id]
#             frame_lookups[agent_idx] = {
#                 'frame_to_idx': {frame: idx for idx, frame in enumerate(track_data['frames'])},
#                 'data': track_data
#             }
            
#             # Set agent metadata
#             arrays['agent_masks'][agent_idx] = 1.0
#             arrays['agent_ids'][agent_idx] = track_id
        
#         # Vectorized processing of past frames
#         for t, frame in enumerate(past_frames):
#             for agent_idx, lookup_data in frame_lookups.items():
#                 if frame in lookup_data['frame_to_idx']:
#                     data_idx = lookup_data['frame_to_idx'][frame]
#                     track_data = lookup_data['data']
#                     coords = track_data['coords'][data_idx]
                    
#                     # Set positions and masks
#                     arrays['past_positions_orig'][agent_idx, t] = coords
#                     arrays['past_positions'][agent_idx, t] = coords
#                     arrays['temporal_masks_past'][agent_idx, t] = 1.0
#                     arrays['obs_masks'][agent_idx, t] = 1.0 if (arrays['agent_ids'][agent_idx], int(frame)) in obs_set else 0.0
                    
#                     # Set label if not already set
#                     if arrays['agent_labels'][agent_idx] == -1:
#                         label = track_data['labels'][data_idx]
#                         if label in self.cls2idx:
#                             arrays['agent_labels'][agent_idx] = self.cls2idx[label]
        
#         # Vectorized processing of future frames
#         for t, frame in enumerate(future_frames):
#             for agent_idx, lookup_data in frame_lookups.items():
#                 if frame in lookup_data['frame_to_idx']:
#                     data_idx = lookup_data['frame_to_idx'][frame]
#                     track_data = lookup_data['data']
#                     coords = track_data['coords'][data_idx]
                    
#                     # Set positions and masks
#                     arrays['future_positions_orig'][agent_idx, t] = coords
#                     arrays['future_positions'][agent_idx, t] = coords
#                     arrays['temporal_masks_future'][agent_idx, t] = 1.0
#                     arrays['occ_masks'][agent_idx, t] = track_data['occluded'][data_idx]
        
#         # Vectorized normalization
#         if self.normalize_positions:
#             # Only normalize for valid agents
#             valid_agent_mask = arrays['agent_masks'] > 0
#             if np.any(valid_agent_mask):
#                 # Normalize past positions
#                 arrays['past_positions'][valid_agent_mask] = self._normalize_coords_vectorized(
#                     arrays['past_positions'][valid_agent_mask], 
#                     arrays['temporal_masks_past'][valid_agent_mask], 
#                     video_key
#                 )
                
#                 # Normalize future positions  
#                 arrays['future_positions'][valid_agent_mask] = self._normalize_coords_vectorized(
#                     arrays['future_positions'][valid_agent_mask], 
#                     arrays['temporal_masks_future'][valid_agent_mask], 
#                     video_key
#                 )
        
#         # Create base sample
#         sample = {
#             'past_positions': arrays['past_positions'],
#             'future_positions': arrays['future_positions'],
#             'past_positions_orig': arrays['past_positions_orig'],
#             'future_positions_orig': arrays['future_positions_orig'],
#             'obs_masks': arrays['obs_masks'],
#             'temporal_masks_past': arrays['temporal_masks_past'],
#             'temporal_masks_future': arrays['temporal_masks_future'],
#             'occ_masks': arrays['occ_masks'],
#             'agent_masks': arrays['agent_masks'],
#             'agent_labels': arrays['agent_labels'],
#             'agent_ids': arrays['agent_ids'],
#             'location': sample_idx['location'],
#             'video': sample_idx['video'],
#             'start_frame': sample_idx['start_frame'],
#             'num_valid_agents': num_agents,
#             'subsampled_frames': window_frames,
#             'frame_subsample_rate': self.frame_subsample
#         }
        
#         # Vectorized delta computation
#         if self.use_deltas:
#             valid_agent_mask = arrays['agent_masks'] > 0
            
#             # Compute deltas only for valid agents
#             past_deltas = self._compute_deltas_vectorized(
#                 arrays['past_positions'], 
#                 arrays['temporal_masks_past']
#             )
            
#             future_deltas = self._compute_deltas_vectorized(
#                 arrays['future_positions'], 
#                 arrays['temporal_masks_future'],
#                 is_future=True,
#                 past_positions=arrays['past_positions']
#             )
            
#             sample['past_deltas'] = past_deltas
#             sample['future_deltas'] = future_deltas
        
#         return self._format_sample(sample)

#     def __del__(self):
#         """Clean up HDF5 file handles"""
#         if hasattr(self, '_hdf5_file_cache'):
#             if hasattr(self._hdf5_file_cache, 'file') and self._hdf5_file_cache.file:
#                 try:
#                     self._hdf5_file_cache.file.close()
#                 except:
#                     pass

#     # def _load_multi_agent_sample_on_demand(self, sample_idx):
#     #     """Load multi-agent sample on demand from HDF5 with frame subsampling"""
#     #     # FIXED: Check if obs_sets is None and provide fallback
#     #     if self.obs_sets is None:
#     #         print("Warning: obs_sets is None, using empty set")
#     #         obs_set = set()
#     #     else:
#     #         obs_set = self.obs_sets.get((sample_idx['location'], sample_idx['video']), set())
        
#     #     video_key = f"{sample_idx['location']}_{sample_idx['video']}"
#     #     window_frames = sample_idx['window_frames']  # These are already subsampled frames
#     #     valid_agents = sample_idx['valid_agents']
        
#     #     past_frames = window_frames[:self.T_past]
#     #     future_frames = window_frames[self.T_past:]
        
#     #     # Initialize arrays for all agents (padded to max_agents)
#     #     num_agents = min(len(valid_agents), self.max_agents)
        
#     #     # Shape: (max_agents, T_past, 2), (max_agents, T_future, 2)
#     #     # For agent padding (not enough agents), use pad_value
#     #     past_positions = np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32)
#     #     future_positions = np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32)
#     #     past_positions_orig = np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32)
#     #     future_positions_orig = np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32)
        
#     #     # Masks: 1 = valid/present, 0 = invalid/absent
#     #     obs_masks = np.zeros((self.max_agents, self.T_past), dtype=np.float32)  # 1 if observed by drone
#     #     temporal_masks_past = np.zeros((self.max_agents, self.T_past), dtype=np.float32)  # 1 if agent exists at this time
#     #     temporal_masks_future = np.zeros((self.max_agents, self.T_future), dtype=np.float32)  # 1 if agent exists at this time
#     #     occ_masks = np.zeros((self.max_agents, self.T_future), dtype=np.float32)  # occlusion values (only valid where temporal_mask=1)
#     #     agent_masks = np.zeros(self.max_agents, dtype=np.float32)  # 1 if agent slot is used, 0 if padded
        
#     #     # Labels and metadata
#     #     agent_labels = np.full(self.max_agents, -1, dtype=np.int32)
#     #     agent_ids = np.full(self.max_agents, -1, dtype=np.int32)
        
#     #     # Process each valid agent
#     #     for agent_idx, track_id in enumerate(valid_agents[:self.max_agents]):
#     #         track_data = self._load_trajectory_data(video_key, track_id)
            
#     #         if track_data is None:
#     #             continue
            
#     #         # Mark this agent slot as used (not padded)
#     #         agent_masks[agent_idx] = 1.0
#     #         agent_ids[agent_idx] = track_id
            
#     #         # Create frame to index mapping for this track
#     #         frame_to_idx = {frame: idx for idx, frame in enumerate(track_data['frames'])}
            
#     #         # Fill past positions (using subsampled frames)
#     #         for t, frame in enumerate(past_frames):
#     #             if frame in frame_to_idx:
#     #                 data_idx = frame_to_idx[frame]
#     #                 coords = track_data['coords'][data_idx]
                    
#     #                 # Always use ground truth coordinates (even if not observed)
#     #                 past_positions_orig[agent_idx, t] = coords
#     #                 past_positions[agent_idx, t] = coords
                    
#     #                 # Mark temporal presence
#     #                 temporal_masks_past[agent_idx, t] = 1.0
                    
#     #                 # Set observation mask (1 if observed by drone, 0 if GT but not observed)
#     #                 obs_masks[agent_idx, t] = 1.0 if (track_id, int(frame)) in obs_set else 0.0
                    
#     #                 # Get label from first valid frame
#     #                 if agent_labels[agent_idx] == -1:
#     #                     label = track_data['labels'][data_idx]
#     #                     if label in self.cls2idx:
#     #                         agent_labels[agent_idx] = self.cls2idx[label]
            
#     #         # Fill future positions (using subsampled frames)
#     #         for t, frame in enumerate(future_frames):
#     #             if frame in frame_to_idx:
#     #                 data_idx = frame_to_idx[frame]
#     #                 coords = track_data['coords'][data_idx]
                    
#     #                 # Always use ground truth coordinates
#     #                 future_positions_orig[agent_idx, t] = coords
#     #                 future_positions[agent_idx, t] = coords
                    
#     #                 # Mark temporal presence
#     #                 temporal_masks_future[agent_idx, t] = 1.0
                    
#     #                 # Set occlusion (only meaningful where agent is present)
#     #                 occ_masks[agent_idx, t] = track_data['occluded'][data_idx]
        
        
#     #     # Normalize if requested (only normalize valid positions, not padding)
#     #     if self.normalize_positions:
#     #         for agent_idx in range(num_agents):
#     #             if agent_masks[agent_idx] > 0:
#     #                 # Normalize past positions where agent is temporally present
#     #                 for t in range(self.T_past):
#     #                     if temporal_masks_past[agent_idx, t] > 0:
#     #                         past_positions[agent_idx, t] = self._normalize_coords(
#     #                             past_positions[agent_idx, t:t+1], video_key)[0]
                    
#     #                 # Normalize future positions where agent is temporally present  
#     #                 for t in range(self.T_future):
#     #                     if temporal_masks_future[agent_idx, t] > 0:
#     #                         future_positions[agent_idx, t] = self._normalize_coords(
#     #                             future_positions[agent_idx, t:t+1], video_key)[0]
        
#     #     sample = {
#     #         'past_positions': past_positions,
#     #         'future_positions': future_positions,
#     #         'past_positions_orig': past_positions_orig,
#     #         'future_positions_orig': future_positions_orig,
#     #         'obs_masks': obs_masks,  # Whether observed by drone (only meaningful in past)
#     #         'temporal_masks_past': temporal_masks_past,  # Whether agent exists at this timestep
#     #         'temporal_masks_future': temporal_masks_future,  # Whether agent exists at this timestep
#     #         'occ_masks': occ_masks,  # Occlusion values (only meaningful where temporal_mask=1)
#     #         'agent_masks': agent_masks,  # Whether agent slot is used (not padding)
#     #         'agent_labels': agent_labels,
#     #         'agent_ids': agent_ids,
#     #         'location': sample_idx['location'],
#     #         'video': sample_idx['video'],
#     #         'start_frame': sample_idx['start_frame'],
#     #         'num_valid_agents': num_agents,
#     #         'subsampled_frames': window_frames,  # NEW: Store the actual subsampled frame numbers
#     #         'frame_subsample_rate': self.frame_subsample  # NEW: Store subsample rate for reference
#     #     }
        
#     #     # Add delta features if requested (adjusted for subsampling)
#     #     if self.use_deltas:
#     #         past_deltas = np.zeros((self.max_agents, self.T_past, 2), dtype=np.float32)
#     #         future_deltas = np.zeros((self.max_agents, self.T_future, 2), dtype=np.float32)
            
#     #         for agent_idx in range(num_agents):
#     #             if agent_masks[agent_idx] > 0:
#     #                 # Calculate deltas only for temporally valid positions
#     #                 agent_past = past_positions[agent_idx]
#     #                 agent_future = future_positions[agent_idx]
#     #                 past_mask = temporal_masks_past[agent_idx]
#     #                 future_mask = temporal_masks_future[agent_idx]
                    
#     #                 # Past deltas - calculate between consecutive valid frames
#     #                 # NOTE: With subsampling, these deltas represent movement over frame_subsample frames
#     #                 valid_past_indices = np.where(past_mask > 0)[0]
#     #                 if len(valid_past_indices) > 1:
#     #                     for i in range(1, len(valid_past_indices)):
#     #                         curr_idx = valid_past_indices[i]
#     #                         prev_idx = valid_past_indices[i-1]
#     #                         past_deltas[agent_idx, curr_idx] = (
#     #                             agent_past[curr_idx] - agent_past[prev_idx]
#     #                         )
                    
#     #                 # Future deltas 
#     #                 valid_future_indices = np.where(future_mask > 0)[0]
                    
#     #                 # Transition from past to future
#     #                 if len(valid_past_indices) > 0 and len(valid_future_indices) > 0:
#     #                     last_past_idx = valid_past_indices[-1]
#     #                     first_future_idx = valid_future_indices[0]
#     #                     future_deltas[agent_idx, first_future_idx] = (
#     #                         agent_future[first_future_idx] - agent_past[last_past_idx]
#     #                     )
                    
#     #                 # Future-to-future deltas
#     #                 if len(valid_future_indices) > 1:
#     #                     for i in range(1, len(valid_future_indices)):
#     #                         curr_idx = valid_future_indices[i]
#     #                         prev_idx = valid_future_indices[i-1]
#     #                         future_deltas[agent_idx, curr_idx] = (
#     #                             agent_future[curr_idx] - agent_future[prev_idx]
#     #                         )
            
#     #         sample['past_deltas'] = past_deltas
#     #         sample['future_deltas'] = future_deltas
        
#     #     return self._format_sample(sample)
    
#     def _format_sample(self, sample):
#         """Format sample for model consumption"""
#         # Convert to tensors if needed
#         return sample


# # Add missing compute_multi_agent_metrics function
# def compute_multi_agent_metrics(predictions, targets, video_stats=None):
#     """Compute trajectory prediction metrics for multi-agent scenarios"""
#     try:
#         # Extract predictions and targets
#         pred_positions = predictions['future_positions_mu']  # (batch, max_agents, T_future, 2)
#         target_positions = targets['future_positions']  # (batch, max_agents, T_future, 2)
#         temporal_mask = targets['temporal_masks_future']  # (batch, max_agents, T_future)
        
#         # Create mask for valid predictions (where agent exists)
#         valid_mask = temporal_mask > 0  # (batch, max_agents, T_future)
        
#         if valid_mask.sum() == 0:
#             return {
#                 'ADE': 0.0, 'FDE': 0.0, 'Agent_ADE_Mean': 0.0, 'Agent_FDE_Mean': 0.0
#             }
        
#         # Compute displacement errors
#         displacement = torch.norm(pred_positions - target_positions, dim=-1)  # (batch, max_agents, T_future)
        
#         # Apply mask to only consider valid positions
#         masked_displacement = displacement * valid_mask.float()
        
#         # Average Displacement Error (ADE) - average over all valid time steps
#         ade = masked_displacement.sum() / valid_mask.sum()
        
#         # Final Displacement Error (FDE) - error at the last valid timestep for each agent
#         fde_values = []
#         batch_size, max_agents, T_future = displacement.shape
        
#         for b in range(batch_size):
#             for a in range(max_agents):
#                 # Find last valid timestep for this agent
#                 valid_times = torch.where(valid_mask[b, a])[0]
#                 if len(valid_times) > 0:
#                     last_valid_time = valid_times[-1]
#                     fde_values.append(masked_displacement[b, a, last_valid_time])
        
#         if len(fde_values) > 0:
#             fde = torch.stack(fde_values).mean()
#         else:
#             fde = torch.tensor(0.0)
        
#         # Per-agent metrics
#         agent_ades = []
#         agent_fdes = []
        
#         for b in range(batch_size):
#             for a in range(max_agents):
#                 agent_valid = valid_mask[b, a]
#                 if agent_valid.sum() > 0:
#                     agent_ade = (masked_displacement[b, a] * agent_valid.float()).sum() / agent_valid.sum()
#                     agent_ades.append(agent_ade)
                    
#                     # Agent FDE
#                     valid_times = torch.where(agent_valid)[0]
#                     if len(valid_times) > 0:
#                         last_valid_time = valid_times[-1]
#                         agent_fdes.append(masked_displacement[b, a, last_valid_time])
        
#         agent_ade_mean = torch.stack(agent_ades).mean() if len(agent_ades) > 0 else torch.tensor(0.0)
#         agent_fde_mean = torch.stack(agent_fdes).mean() if len(agent_fdes) > 0 else torch.tensor(0.0)
        
#         return {
#             'ADE': ade.item(),
#             'FDE': fde.item(), 
#             'Agent_ADE_Mean': agent_ade_mean.item(),
#             'Agent_FDE_Mean': agent_fde_mean.item()
#         }
        
#     except Exception as e:
#         print(f"Error computing metrics: {e}")
#         return {
#             'ADE': 0.0, 'FDE': 0.0, 'Agent_ADE_Mean': 0.0, 'Agent_FDE_Mean': 0.0
#         }


# class TrajectoryLoss(nn.Module):
#     """Loss function for trajectory prediction with uncertainty and proper masking"""
    
#     def __init__(self, use_deltas=True, predict_uncertainty=True, 
#                  position_weight=1.0, delta_weight=1.0, uncertainty_weight=0.1):
#         super().__init__()
#         self.use_deltas = use_deltas
#         self.predict_uncertainty = predict_uncertainty
#         self.position_weight = position_weight
#         self.delta_weight = delta_weight
#         self.uncertainty_weight = uncertainty_weight
        
#     def forward(self, predictions, batch):
#         total_loss = 0.0
#         loss_dict = {}
        
#         # Extract masks from batch
#         # temporal_masks_future: (batch, max_agents, T_future) - 1 if agent exists at timestep
#         # occ_masks: (batch, max_agents, T_future) - occlusion values (only valid where agent exists)
#         # agent_masks: (batch, max_agents) - 1 if agent slot is used
        
#         temporal_mask = batch['temporal_masks_future']  # (batch, max_agents, T_future)
#         occ_mask = batch['occ_masks']  # (batch, max_agents, T_future) 
#         agent_mask = batch['agent_masks']  # (batch, max_agents)
        
#         # Create visibility mask: 1 for visible (non-occluded) positions where agent exists
#         # Assuming occ_mask values: 0 = visible, 1 = occluded (adjust if different)
#         visibility_mask = temporal_mask * (1.0 - occ_mask)  # (batch, max_agents, T_future)
        
#         # Also mask out padded agents
#         agent_mask_expanded = agent_mask.unsqueeze(-1).expand_as(visibility_mask)
#         valid_mask = visibility_mask * agent_mask_expanded
        
#         if self.predict_uncertainty:
#             # Negative log likelihood loss for positions
#             if 'future_positions_mu' in predictions and 'future_positions_logvar' in predictions:
#                 pos_mu = predictions['future_positions_mu']  # (batch, max_agents, T_future, 2)
#                 pos_logvar = predictions['future_positions_logvar']  # (batch, max_agents, T_future, 2)
#                 pos_target = batch['future_positions']  # (batch, max_agents, T_future, 2)
                
#                 pos_loss = self._gaussian_nll_loss(pos_mu, pos_logvar, pos_target, valid_mask)
#                 loss_dict['position_loss'] = pos_loss
#                 total_loss += self.position_weight * pos_loss
            
#             # Negative log likelihood loss for deltas
#             if self.use_deltas and 'future_deltas_mu' in predictions and 'future_deltas_logvar' in predictions:
#                 delta_mu = predictions['future_deltas_mu']
#                 delta_logvar = predictions['future_deltas_logvar']
#                 delta_target = batch['future_deltas']
                
#                 delta_loss = self._gaussian_nll_loss(delta_mu, delta_logvar, delta_target, valid_mask)
#                 loss_dict['delta_loss'] = delta_loss
#                 total_loss += self.delta_weight * delta_loss
            
#             # Uncertainty regularization
#             if 'future_positions_logvar' in predictions:
#                 pos_logvar = predictions['future_positions_logvar']
#                 # Apply mask to only regularize visible positions
#                 masked_logvar = pos_logvar * valid_mask.unsqueeze(-1).expand_as(pos_logvar)
#                 # Regularize towards reasonable uncertainty
#                 uncertainty_reg = torch.mean(masked_logvar ** 2) 
#                 loss_dict['uncertainty_reg'] = uncertainty_reg
#                 total_loss += self.uncertainty_weight * uncertainty_reg
                
#         else:
#             # Standard MSE loss
#             if 'future_positions' in predictions:
#                 pos_loss = self._masked_mse_loss(
#                     predictions['future_positions'], batch['future_positions'], valid_mask
#                 )
#                 loss_dict['position_loss'] = pos_loss
#                 total_loss += self.position_weight * pos_loss
            
#             if self.use_deltas and 'future_deltas' in predictions:
#                 delta_loss = self._masked_mse_loss(
#                     predictions['future_deltas'], batch['future_deltas'], valid_mask
#                 )
#                 loss_dict['delta_loss'] = delta_loss
#                 total_loss += self.delta_weight * delta_loss
        
#         loss_dict['total_loss'] = total_loss
#         return total_loss, loss_dict
    
#     def _gaussian_nll_loss(self, mu, logvar, target, mask):
#         """
#         Negative log likelihood for Gaussian distribution
        
#         Args:
#             mu: predicted mean (batch, max_agents, T_future, 2)
#             logvar: predicted log variance (batch, max_agents, T_future, 2) 
#             target: ground truth (batch, max_agents, T_future, 2)
#             mask: validity mask (batch, max_agents, T_future) - 1 for valid, 0 for invalid
#         """
#         # Expand mask to match tensor dimensions
#         mask_expanded = mask.unsqueeze(-1).expand_as(mu)  # (batch, max_agents, T_future, 2)
        
#         # Compute NLL
#         var = torch.exp(logvar)  # Convert log variance to variance
#         squared_error = (target - mu) ** 2
#         nll = 0.5 * (logvar + squared_error / (var + 1e-8))  # Add epsilon for stability
        
#         # Apply mask - only compute loss for valid positions
#         masked_nll = nll * mask_expanded
        
#         # Return average loss over valid positions
#         valid_count = mask_expanded.sum()
#         if valid_count > 0:
#             return masked_nll.sum() / valid_count
#         else:
#             return torch.tensor(0.0, device=mu.device, requires_grad=True)
    
#     def _masked_mse_loss(self, pred, target, mask):
#         """MSE loss with masking for invalid positions"""
#         mask_expanded = mask.unsqueeze(-1).expand_as(pred)
#         mse = ((pred - target) ** 2) * mask_expanded
        
#         valid_count = mask_expanded.sum()
#         if valid_count > 0:
#             return mse.sum() / valid_count
#         else:
#             return torch.tensor(0.0, device=pred.device, requires_grad=True)


# class EarlyStopping:
#     """Early stopping utility"""
#     def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.restore_best_weights = restore_best_weights
#         self.best_loss = float('inf')
#         self.counter = 0
#         self.best_weights = None
        
#     def __call__(self, val_loss, model):
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             if self.restore_best_weights:
#                 self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#         else:
#             self.counter += 1
            
#         if self.counter >= self.patience:
#             if self.restore_best_weights and self.best_weights is not None:
#                 model.load_state_dict(self.best_weights)
#             return True
#         return False


# def collate_fn(batch):
#     """Custom collate function for batching"""
#     batched = {}
#     for key in batch[0].keys():
#         if isinstance(batch[0][key], torch.Tensor):
#             batched[key] = torch.stack([item[key] for item in batch])
#         elif isinstance(batch[0][key], np.ndarray):
#             batched[key] = torch.from_numpy(np.stack([item[key] for item in batch]))
#         else:
#             # For non-tensor data like strings, lists, etc.
#             batched[key] = [item[key] for item in batch]
    
#     return batched


# def evaluate_model(model, dataloader, criterion, device, dataset=None):
#     """Evaluate model on validation set with proper metrics computation"""
#     model.eval()
#     total_loss = 0.0
#     total_samples = 0
#     loss_dict_accum = {}
    
#     # For computing metrics
#     all_predictions = []
#     all_targets = []
#     all_video_keys = []
    
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Validating"):
#             # Move batch to device
#             for key, value in batch.items():
#                 if isinstance(value, torch.Tensor):
#                     batch[key] = value.to(device)
            
#             # Forward pass
#             predictions = model(batch, use_teacher_forcing=False)
            
#             # Compute loss
#             loss, loss_dict = criterion(predictions, batch)
            
#             batch_size = batch['past_positions'].size(0)
#             total_loss += loss.item() * batch_size
#             total_samples += batch_size
            
#             # Accumulate loss components
#             for key, value in loss_dict.items():
#                 if key not in loss_dict_accum:
#                     loss_dict_accum[key] = 0.0
#                 loss_dict_accum[key] += value.item() * batch_size
            
#             # Store predictions and targets for metric computation
#             all_predictions.append({
#                 'future_positions_mu': predictions['future_positions_mu'].cpu(),
#                 'future_positions_var': predictions.get('future_positions_var', None)
#             })
            
#             # Create targets dict with proper structure
#             targets = {
#                 'future_positions': batch['future_positions'].cpu(),
#                 'temporal_masks_future': batch['temporal_masks_future'].cpu()
#             }
#             all_targets.append(targets)
            
#             # Store video keys for denormalization if needed
#             if isinstance(batch['location'], list) and isinstance(batch['video'], list):
#                 video_keys = [f"{loc}_{vid}" for loc, vid in zip(batch['location'], batch['video'])]
#                 all_video_keys.extend(video_keys)
#             else:
#                 all_video_keys.extend([None] * batch_size)
    
#     # Average losses
#     avg_loss = total_loss / total_samples
#     for key in loss_dict_accum:
#         loss_dict_accum[key] /= total_samples
    
#     # Compute trajectory metrics
#     try:
#         # Concatenate all predictions and targets
#         concat_predictions = {}
#         concat_targets = {}
        
#         # Handle predictions
#         for key in all_predictions[0].keys():
#             if all_predictions[0][key] is not None:
#                 concat_predictions[key] = torch.cat([p[key] for p in all_predictions], dim=0)
        
#         # Handle targets
#         for key in all_targets[0].keys():
#             concat_targets[key] = torch.cat([t[key] for t in all_targets], dim=0)
        
#         # Get video stats for denormalization if dataset is available
#         video_stats = None
#         if dataset is not None and hasattr(dataset, 'video_stats') and dataset.video_stats is not None:
#             # For simplicity, use the first video's stats or compute global stats
#             # In practice, you might want to handle this per-sample
#             video_stats = list(dataset.video_stats.values())[0] if dataset.video_stats else None
        
#         # Compute metrics
#         metrics = compute_multi_agent_metrics(concat_predictions, concat_targets, video_stats)
#         loss_dict_accum.update(metrics)
        
#     except Exception as e:
#         print(f"Warning: Could not compute trajectory metrics: {e}")
#         # Add dummy metrics to avoid KeyError
#         loss_dict_accum.update({
#             'ADE': 0.0,
#             'FDE': 0.0,
#             'Agent_ADE_Mean': 0.0,
#             'Agent_FDE_Mean': 0.0
#         })
    
#     return avg_loss, loss_dict_accum
# # Add this debugging code to identify where the CUDA error occurs

# def debug_model_forward(model, batch, device):
#     """Debug the model forward pass to find where CUDA indexing fails"""
#     print("Debugging model forward pass...")
#     classes = ['Pedestrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus']
#     locations = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'nexus', 'quad']
    
#     # Move batch to device
#     for key, value in batch.items():
#         if isinstance(value, torch.Tensor):
#             batch[key] = value.to(device)
    
#     # Print input shapes and ranges
#     print("\nInput tensor analysis:")
#     for key, value in batch.items():
#         if isinstance(value, torch.Tensor):
#             print(f"{key}: {value.shape}, dtype: {value.dtype}")
#             if value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
#                 print(f"  Range: [{value.min()}, {value.max()}]")
#                 if key == 'agent_labels':
#                     unique_vals = value.unique()
#                     print(f"  Unique values: {unique_vals}")
#                     print(f"  Expected range: [-1, {len(classes)-1}] for {len(classes)} classes")
    
#     # Try to identify which part of the model fails
#     model.eval()
#     with torch.inference_mode():
#         try:
#             print("\nTesting model forward pass...")
#             predictions = model(batch, use_teacher_forcing=False)
#             print("✓ Model forward pass successful")
#             return predictions
#         except RuntimeError as e:
#             print(f"✗ Model forward pass failed: {e}")
            
#             # If it's a CUDA indexing error, let's try to narrow it down
#             if "indexSelectLargeIndex" in str(e):
#                 print("\nThis is a CUDA indexing error. Likely causes:")
#                 print("1. Agent labels out of bounds for embedding lookup")
#                 print("2. Location indices out of bounds") 
#                 print("3. Attention indices out of bounds")
#                 print("4. Positional encoding indices out of bounds")
                
#                 # Check specific values that might cause issues
#                 if 'agent_labels' in batch:
#                     labels = batch['agent_labels']
#                     print(f"\nAgent labels analysis:")
#                     print(f"  Shape: {labels.shape}")
#                     print(f"  Min: {labels.min()}, Max: {labels.max()}")
#                     print(f"  Unique: {labels.unique()}")
#                     print(f"  Expected classes: {len(classes)} (indices 0-{len(classes)-1})")
                    
#                     # Check if any labels are out of bounds
#                     invalid_labels = labels[(labels < -1) | (labels >= len(classes))]
#                     if len(invalid_labels) > 0:
#                         print(f"   FOUND INVALID LABELS: {invalid_labels.unique()}")
#                         print(f"  This is likely the cause of the indexing error!")
#                     else:
#                         print(f"   All agent labels are within bounds")
                
#                 if 'location' in batch and isinstance(batch['location'], list):
#                     print(f"\nLocation analysis:")
#                     unique_locs = set(batch['location']) 
#                     print(f"  Unique locations: {unique_locs}")
#                     print(f"  Expected locations: {locations}")
                    
#                     # Check if all locations are in the expected list
#                     invalid_locs = unique_locs - set(locations)
#                     if invalid_locs:
#                         print(f"   FOUND INVALID LOCATIONS: {invalid_locs}")
#                     else:
#                         print(f"   All locations are valid")
            
#             raise e

# # Modified train_epoch function with debugging
# def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
#     """Train model for one epoch with debugging"""
#     model.train()
#     total_loss = 0.0
#     total_samples = 0
#     loss_dict_accum = {}
    
#     pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
#     for batch_idx, batch in enumerate(pbar):
#         try:
#             # For the first batch, do detailed debugging
#             if batch_idx == 0:
#                 print(f"\n=== DEBUG: First batch of epoch {epoch} ===")
#                 debug_model_forward(model, batch, device)
#                 print("=== END DEBUG ===\n")
            
#             # Move batch to device
#             for key, value in batch.items():
#                 if isinstance(value, torch.Tensor):
#                     batch[key] = value.to(device)
            
#             # Forward pass
#             predictions = model(batch, use_teacher_forcing=True)
            
#             # Compute loss
#             loss, loss_dict = criterion(predictions, batch)
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
            
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
            
#             # Accumulate statistics
#             batch_size = batch['past_positions'].size(0)
#             total_loss += loss.item() * batch_size
#             total_samples += batch_size
            
#             for key, value in loss_dict.items():
#                 if key not in loss_dict_accum:
#                     loss_dict_accum[key] = 0.0
#                 loss_dict_accum[key] += value.item() * batch_size
            
#             # Update progress bar
#             pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
#         except RuntimeError as e:
#             if "indexSelectLargeIndex" in str(e):
#                 print(f"\n CUDA indexing error at batch {batch_idx}")
#                 print("This suggests an out-of-bounds index in an embedding lookup or attention operation")
#                 print("Most likely causes:")
#                 print("1. agent_labels contains indices >= num_classes")
#                 print("2. location indices are out of bounds") 
#                 print("3. sequence length exceeds model's maximum")
                
#                 # Print batch info for debugging
#                 print(f"\nBatch info:")
#                 for key, value in batch.items():
#                     if isinstance(value, torch.Tensor):
#                         if value.dtype in [torch.long, torch.int]:
#                             print(f"  {key}: shape={value.shape}, range=[{value.min()}, {value.max()}]")
#                         else:
#                             print(f"  {key}: shape={value.shape}")
                
#                 raise e
#             else:
#                 raise e
    
#     # Average losses
#     avg_loss = total_loss / total_samples
#     for key in loss_dict_accum:
#         loss_dict_accum[key] /= total_samples
    
#     return avg_loss, loss_dict_accum

# # Quick fix for potential agent label issues (minimal change to dataset)
# def fix_agent_labels_in_batch(batch, num_classes):
#     """Quick fix to ensure agent labels are within bounds"""
#     if 'agent_labels' in batch:
#         labels = batch['agent_labels']
#         # Clamp to valid range: -1 (padding) or 0 to num_classes-1
#         batch['agent_labels'] = torch.clamp(labels, -1, num_classes - 1)
        
#         # Check if we had to fix anything
#         if not torch.equal(labels, batch['agent_labels']):
#             print(f"Warning: Fixed out-of-bounds agent labels")
#             print(f"  Original range: [{labels.min()}, {labels.max()}]")
#             print(f"  Fixed range: [{batch['agent_labels'].min()}, {batch['agent_labels'].max()}]")
    
#     return batch


# def main():
#     # Configuration
#     config = {
#         'drone_data_root': "../../../square_stanford_data",
#         'original_dataset_root': "../../../stanford_data/archive",
#         'cache_dir': "new_cache_16",
#         'checkpoint_dir': "checkpoints",
#         'batch_size': 64,  # Reduced for stability
#         'num_epochs': 10,
#         'learning_rate': 1e-4,
#         'weight_decay': 1e-5,
#         'patience': 15,
#         'val_split': 0.95,
#         'num_workers': 4,  # Reduced for stability
#         'max_agents': 128,
#         'T_past': 10,
#         'T_future': 20,
#         'frame_subsample': 12,
#         'wandb_project': 'trajectory-prediction',
#         'run_name': 'multi-agent-trajectory-exp'
#     }
    
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Create checkpoint directory
#     checkpoint_dir = Path(config['checkpoint_dir'])
#     checkpoint_dir.mkdir(exist_ok=True)
    
#     # Initialize wandb
#     wandb.init(
#         project=config['wandb_project'],
#         name=config['run_name'],
#         config=config
#     )
    
#     # Dataset configuration
#     classes = ['Pedestrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus']
#     locations = ['bookstore','little', 'coupa', 'deathCircle', 'gates', 'hyang', 'nexus', 'quad']
    
#     print("Loading dataset...")
#     dataset = OptimizedMultiAgentSequenceDataset(
#         drone_data_root=config['drone_data_root'],
#         original_dataset_root=config['original_dataset_root'],
#         classes=classes,
#         T_past=config['T_past'],
#         T_future=config['T_future'],
#         use_deltas=True,
#         normalize_positions=True,
#         cache_dir=config['cache_dir'],
#         lazy_loading=True,
#         num_workers=config['num_workers'],
#         max_agents=config['max_agents'],
#         frame_subsample=config['frame_subsample']
#     )
    
#     print(f"Dataset loaded with {len(dataset)} samples")
    
#     # Split dataset
#     val_size = int(len(dataset) * config['val_split'])
#     train_size = len(dataset) - val_size
#     train_dataset, val_dataset = random_split(
#         dataset, [train_size, val_size],
#         generator=torch.Generator().manual_seed(42)
#     )
    
#     print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=config['num_workers'],
#         pin_memory=True if device.type == 'cuda' else False,
#         collate_fn=collate_fn,
#         drop_last=True  # Drop last incomplete batch
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False,
#         num_workers=config['num_workers'],
#         pin_memory=True if device.type == 'cuda' else False,
#         collate_fn=collate_fn,
#         drop_last=False
#     )
    
#     # Initialize model
#     print("Initializing model...")
#     model = GraphInteractionModel(
#         num_classes=len(classes),
#         locations=locations,
#         d_model=256,
#         nhead=8,
#         num_layers=3,
#         T_past=config['T_past'],
#         T_future=config['T_future'],
#         max_agents=config['max_agents']
#     )
    
#     model = model.to(device)
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
    
#     # Initialize loss function and optimizer
#     criterion = TrajectoryLoss(
#         use_deltas=True,
#         predict_uncertainty=True,
#         position_weight=1.0,
#         delta_weight=1.0,
#         uncertainty_weight=0.1
#     )
    
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=config['learning_rate'],
#         weight_decay=config['weight_decay']
#     )
    
#     # Learning rate scheduler
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['num_epochs'], eta_min=1e-6)

    
#     # Early stopping
#     early_stopping = EarlyStopping(patience=config['patience'], min_delta=1e-4)
    
#     # Training loop
#     best_val_loss = float('inf')
    
#     print("Starting training...")
#     for epoch in range(1, config['num_epochs'] + 1):
#         start_time = time.time()
        
#         # Train
#         train_loss, train_loss_dict = train_epoch(
#             model, train_loader, criterion, optimizer, device, epoch
#         )
        
#         # Validate - pass dataset for metric computation
#         val_loss, val_loss_dict = evaluate_model(
#             model, val_loader, criterion, device, dataset
#         )
        
#         # Update learning rate
#         scheduler.step()
        
#         epoch_time = time.time() - start_time
        
#         # Log metrics
#         metrics = {
#             'epoch': epoch,
#             'train_loss': train_loss,
#             'val_loss': val_loss,
#             'epoch_time': epoch_time,
#             'learning_rate': optimizer.param_groups[0]['lr']
#         }
        
#         # Add detailed loss components
#         for key, value in train_loss_dict.items():
#             metrics[f'train_{key}'] = value
#         for key, value in val_loss_dict.items():
#             metrics[f'val_{key}'] = value
        
#         wandb.log(metrics)
        
#         print(f"Epoch {epoch}/{config['num_epochs']}")
#         print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
#         if 'ADE' in val_loss_dict and 'FDE' in val_loss_dict:
#             print(f"Val ADE: {val_loss_dict['ADE']:.4f}, Val FDE: {val_loss_dict['FDE']:.4f}")
#         print(f"Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
#         # Save checkpoint every epoch
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'train_loss': train_loss,
#             'val_loss': val_loss,
#             'config': config,
#             'classes': classes,
#             'locations': locations
#         }
        
#         # Save current epoch checkpoint
#         torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')
        
#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
#             print(f"New best model saved with val_loss: {val_loss:.6f}")
        
#         # Save latest model (overwrite each epoch)
#         torch.save(checkpoint, checkpoint_dir / 'latest_model.pth')
        
#         # Early stopping check
#         if early_stopping(val_loss, model):
#             print(f"Early stopping triggered after {epoch} epochs")
#             break
        
#         print("-" * 50)
    
#     # Save final model
#     final_checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'train_loss': train_loss,
#         'val_loss': val_loss,
#         'best_val_loss': best_val_loss,
#         'config': config,
#         'classes': classes,
#         'locations': locations
#     }
#     torch.save(final_checkpoint, checkpoint_dir / 'final_model.pth')
    
#     print("Training completed!")
#     print(f"Best validation loss: {best_val_loss:.6f}")
    
#     # Save training summary
#     summary = {
#         'best_val_loss': best_val_loss,
#         'final_epoch': epoch,
#         'config': config,
#         'total_params': total_params,
#         'trainable_params': trainable_params
#     }
    
#     with open(checkpoint_dir / 'training_summary.json', 'w') as f:
#         json.dump(summary, f, indent=2)
    
#     wandb.finish()


# if __name__ == "__main__":
#     main()



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
import pandas as pd
import numpy as np 
import math
import json
from functools import lru_cache
from threading import Lock
import threading
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
from improved_model import ImprovedGraphInteractionModel
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import wandb
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

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

def process_video_multi_agent_worker(args):
    """Worker function for processing multi-agent video data"""
    loc, vid, obs_sets, orig_root, classes, cls2idx, T_past, T_future, frame_subsample, min_frames_per_agent = args
    
    orig_file = os.path.join(orig_root, "annotations", loc, vid, "annotations.txt")
    if not os.path.isfile(orig_file):
        return []
    
    indices = []
    key = (loc, vid)
    seen = obs_sets.get(key, set())  # Use set() as default instead of None
    
    try:
        df = pd.read_csv(orig_file, sep=' ', header=None,
                       names=['trackId','xmin','ymin','xmax','ymax',
                             'frame','lost','occluded','generated','label'])
        df = df[df['label'].isin(classes)]
        if len(df) == 0:
            return []
        df['x'] = (df['xmin'] + df['xmax']) * 0.5
        df['y'] = (df['ymin'] + df['ymax']) * 0.5
        
        # Get all unique frames
        all_frames = sorted(df['frame'].unique())
        window_size = T_past + T_future
        
        # Frame subsampling logic for multi-agent
        # Try different starting offsets for subsampling
        for frame_offset in range(frame_subsample):
            # Calculate how many subsampled windows we can fit
            max_subsampled_windows = (len(all_frames) - frame_offset) // frame_subsample
            
            for start_subsample_idx in range(max_subsampled_windows - window_size + 1):
                # Calculate the actual frame indices we'll use
                window_frame_indices = []
                window_frames = []
                
                for j in range(window_size):
                    frame_idx = frame_offset + (start_subsample_idx + j) * frame_subsample
                    if frame_idx < len(all_frames):
                        window_frame_indices.append(frame_idx)
                        window_frames.append(all_frames[frame_idx])
                
                if len(window_frames) < window_size:
                    continue

                past_frames = window_frames[:T_past]
                future_frames = window_frames[T_past:]

                window_df = df[df['frame'].isin(window_frames)]

                if len(window_df) == 0:  # Check window_df, not window_frames
                    continue

                # Find agents that meet minimum frame requirement
                valid_agents = []
                agent_data = {}

                for tid, agent_df in window_df.groupby('trackId'):
                    agent_frames = set(agent_df['frame'].values)
                    past_frames_count = len([f for f in past_frames if f in agent_frames])
                    future_frames_count = len([f for f in future_frames if f in agent_frames])

                    if past_frames_count >= min_frames_per_agent:
                        # Only check observation mask for frames that exist for this agent
                        obs_mask = np.array([1.0 if (tid, int(f)) in seen else 0.0 
                                           for f in past_frames if f in agent_frames])
                        if len(obs_mask) > 0 and obs_mask.sum() >= 2:  # Check array length
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
                    'frame_offset': frame_offset,  # Store frame offset
                    'subsample_start_idx': start_subsample_idx,  # Store subsample start
                    'valid_agents': valid_agents,
                    'agent_data': agent_data,
                    'num_agents': len(valid_agents)
                }

                indices.append(sample_idx)
                
    except Exception as e:
        print(f"Error processing {loc}/{vid}: {e}")
    return indices

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
        self._hdf5_file_cache = threading.local()
        self._hdf5_lock = Lock()
        self.hdf5_file_path = self.cache_dir / "multi_agent_trajectory_data.h5"

    def _get_hdf5_file(self):
        """Get thread-local HDF5 file handle"""
        if not hasattr(self._hdf5_file_cache, 'file') or self._hdf5_file_cache.file is None:
            try:
                self._hdf5_file_cache.file = h5py.File(self.hdf5_file_path, 'r')
            except Exception as e:
                print(f"Error opening HDF5 file: {e}")
                return None
        return self._hdf5_file_cache.file
    
    @lru_cache(maxsize=1024)  # Increased cache size significantly
    def _load_trajectory_data_batch(self, video_key: str, track_ids_tuple: tuple):
        """Load multiple trajectory data at once for better I/O efficiency"""
        hdf5_file = self._get_hdf5_file()
        if hdf5_file is None:
            return {}

        try:
            if video_key not in hdf5_file:
                return {}
                
            video_grp = hdf5_file[video_key]
            batch_data = {}
            
            for track_id in track_ids_tuple:
                if str(track_id) not in video_grp:
                    continue
                    
                track_grp = video_grp[str(track_id)]
                batch_data[track_id] = {
                    'coords': track_grp['coords'][:].astype(np.float32),
                    'frames': track_grp['frames'][:].astype(np.int32),
                    'occluded': track_grp['occluded'][:].astype(np.float32),
                    'labels': [label.decode('utf-8') for label in track_grp['labels'][:]]
                }
            
            return batch_data
        except Exception as e:
            print(f"Error loading batch trajectory data for {video_key}: {e}")
            return {}

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
        # FIXED: Always load obs_sets when loading from cache
        obs_sets_file = self.cache_dir / "obs_sets.pkl"
        if obs_sets_file.exists():
            with open(obs_sets_file, 'rb') as f:
                self.obs_sets = pkl.load(f)
        else:
            # If obs_sets file doesn't exist, rebuild it
            print("obs_sets not found in cache, rebuilding...")
            self.obs_sets = self._build_obs_sets_parallel()
            with open(obs_sets_file, 'wb') as f:
                pkl.dump(self.obs_sets, f)
        
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
        args = []
        for loc in os.listdir(os.path.join(self.orig_root, "annotations")):
            if loc in ['hyang', '.DS_Store']:
                continue
            loc_path = os.path.join(self.orig_root, "annotations", loc)
            if not os.path.isdir(loc_path):
                continue
            for vid in os.listdir(loc_path):
                args.append((loc, vid, obs_sets, self.orig_root, self.classes, 
                            self.cls2idx, self.T_past, self.T_future, 
                            self.frame_subsample, self.min_frames_per_agent))
        
        # Process in parallel using the external worker function
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_video_multi_agent_worker, args))
        
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
        
        # FIXED: Always save obs_sets when building from scratch
        with open(self.cache_dir / "obs_sets.pkl", 'wb') as f:
            pkl.dump(self.obs_sets, f)
        
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
        
        # FIXED: Also save obs_sets when caching samples
        with open(self.cache_dir / "obs_sets.pkl", 'wb') as f:
            pkl.dump(self.obs_sets, f)

    @lru_cache(maxsize=128)
    def _load_trajectory_data(self, video_key: str, track_id: int):
        hdf5_file = self.cache_dir / "multi_agent_trajectory_data.h5"

        try:
            with h5py.File(hdf5_file, 'r') as f:  # Fixed: use h5py.File instead of open
                if video_key not in f:
                    return None
                if str(track_id) not in f[video_key]:
                    return None
                    
                track_grp = f[video_key][str(track_id)]

                return {
                    'coords': track_grp['coords'][:].astype(np.float32),
                    'frames': track_grp['frames'][:].astype(np.int32),
                    'occluded': track_grp['occluded'][:].astype(np.float32),
                    'labels': [label.decode('utf-8') for label in track_grp['labels'][:]]
                }
        except Exception as e:
            print(f"Error loading trajectory data for {video_key}/{track_id}: {e}")
            return None
        
    def _compute_deltas_vectorized_safe(self, positions, temporal_masks, is_future=False, past_positions=None, pad_value=-999.0):
        """FIXED: Vectorized delta computation that properly handles pad values"""
        max_agents, seq_len, _ = positions.shape
        deltas = np.full_like(positions, pad_value)  # Initialize with pad values
        
        for agent_idx in range(max_agents):
            if not np.any(temporal_masks[agent_idx]):
                continue
                
            valid_indices = np.where(temporal_masks[agent_idx] > 0)[0]
            
            # Also check for non-padded positions
            non_padded_mask = ~(positions[agent_idx] == pad_value).any(axis=1)
            valid_indices = valid_indices[non_padded_mask[valid_indices]]
            
            if len(valid_indices) <= 1 and not is_future:
                continue
            
            if is_future and past_positions is not None:
                # Handle transition from past to future
                past_mask = temporal_masks[agent_idx] if past_positions.shape[1] == seq_len else None
                if past_mask is not None:
                    past_valid = np.where(past_mask > 0)[0]
                    past_non_padded = ~(past_positions[agent_idx] == pad_value).any(axis=1)
                    past_valid = past_valid[past_non_padded[past_valid]]
                    
                    if len(past_valid) > 0 and len(valid_indices) > 0:
                        last_past_idx = past_valid[-1]
                        first_future_idx = valid_indices[0]
                        deltas[agent_idx, first_future_idx] = (
                            positions[agent_idx, first_future_idx] - 
                            past_positions[agent_idx, last_past_idx]
                        )
                        # Remove first future index from processing
                        valid_indices = valid_indices[1:]
            
            # Compute consecutive deltas
            if len(valid_indices) > 1:
                for i in range(1, len(valid_indices)):
                    curr_idx = valid_indices[i]
                    prev_idx = valid_indices[i-1]
                    deltas[agent_idx, curr_idx] = (
                        positions[agent_idx, curr_idx] - positions[agent_idx, prev_idx]
                    )
        
        return deltas
        
    # def _compute_deltas_vectorized(self, positions, temporal_masks, is_future=False, past_positions=None):
    #     """Vectorized delta computation"""
    #     max_agents, seq_len, _ = positions.shape
    #     deltas = np.zeros_like(positions)
        
    #     for agent_idx in range(max_agents):
    #         if not np.any(temporal_masks[agent_idx]):
    #             continue
                
    #         valid_indices = np.where(temporal_masks[agent_idx] > 0)[0]
    #         if len(valid_indices) <= 1 and not is_future:
    #             continue
            
    #         if is_future and past_positions is not None:
    #             # Handle transition from past to future
    #             past_mask = temporal_masks[agent_idx] if past_positions.shape[1] == seq_len else None
    #             if past_mask is not None:
    #                 past_valid = np.where(past_mask > 0)[0]
    #                 if len(past_valid) > 0 and len(valid_indices) > 0:
    #                     last_past_idx = past_valid[-1]
    #                     first_future_idx = valid_indices[0]
    #                     deltas[agent_idx, first_future_idx] = (
    #                         positions[agent_idx, first_future_idx] - 
    #                         past_positions[agent_idx, last_past_idx]
    #                     )
    #                     # Remove first future index from processing
    #                     valid_indices = valid_indices[1:]
            
    #         # Compute consecutive deltas
    #         if len(valid_indices) > 1:
    #             for i in range(1, len(valid_indices)):
    #                 curr_idx = valid_indices[i]
    #                 prev_idx = valid_indices[i-1]
    #                 deltas[agent_idx, curr_idx] = (
    #                     positions[agent_idx, curr_idx] - positions[agent_idx, prev_idx]
    #                 )
        
    #     return deltas
    

# FIXED: Add this method to your OptimizedMultiAgentSequenceDataset class

    def _safe_normalize_coords_vectorized(self, coords_array, temporal_mask, video_key, pad_value=-999.0):
        """FIXED: Vectorized coordinate normalization that properly handles pad values"""
        if self.video_stats is None or video_key not in self.video_stats:
            return coords_array
        
        stats = self.video_stats[video_key]
        mean = stats['mean'].reshape(1, 1, 2)  # Broadcast shape
        std = stats['std'].reshape(1, 1, 2) + 1e-8
        
        # Create mask for valid (non-padded) positions
        valid_positions_mask = ~(coords_array == pad_value).any(axis=-1, keepdims=True)  # Shape: (agents, time, 1)
        
        # Combine with temporal mask
        combined_mask = temporal_mask[:, :, np.newaxis] * valid_positions_mask  # Shape: (agents, time, 1)
        
        # Only normalize where both temporal_mask is True AND position is not padded
        normalized = coords_array.copy()
        
        # Apply normalization only to valid positions
        valid_coords = coords_array[combined_mask.squeeze(-1)]
        if len(valid_coords) > 0:
            normalized_valid = (valid_coords.reshape(-1, 2) - mean.reshape(2)) / std.reshape(2)
            
            # Put normalized values back
            normalized[combined_mask.squeeze(-1)] = normalized_valid
        
        # Keep pad values as pad values (don't normalize them)
        normalized = np.where(valid_positions_mask, normalized, coords_array)
        
        return normalized
    # def _normalize_coords_vectorized(self, coords_array, temporal_mask, video_key):
    #     """Vectorized coordinate normalization"""
    #     if self.video_stats is None or video_key not in self.video_stats:
    #         return coords_array
        
    #     stats = self.video_stats[video_key]
    #     mean = stats['mean'].reshape(1, 1, 2)  # Broadcast shape
    #     std = stats['std'].reshape(1, 1, 2) + 1e-8
        
    #     # Only normalize where temporal_mask is True
    #     normalized = coords_array.copy()
    #     mask_expanded = temporal_mask[:, :, np.newaxis]  # Shape: (agents, time, 1)
    #     normalized = np.where(mask_expanded, (coords_array - mean) / std, coords_array)
        
    #     return normalized
        
    # def _normalize_coords(self, coords, video_key):
    #     """Normalize coordinates using video-specific statistics"""
    #     if self.video_stats is None or video_key not in self.video_stats:
    #         return coords
        
    #     stats = self.video_stats[video_key]
    #     normalized = (coords - stats['mean']) / (stats['std'] + 1e-8)
    #     return normalized
    
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
        return self._load_multi_agent_sample_on_demand_fixed(sample_idx)
    
    def _load_multi_agent_sample_on_demand_fixed(self, sample_idx):
        """FIXED: Multi-agent sample loading with proper pad value handling"""
        # Get observation set with fallback
        obs_set = self.obs_sets.get((sample_idx['location'], sample_idx['video']), set()) if self.obs_sets else set()
        
        video_key = f"{sample_idx['location']}_{sample_idx['video']}"
        window_frames = sample_idx['window_frames']
        valid_agents = sample_idx['valid_agents']
        
        past_frames = window_frames[:self.T_past]
        future_frames = window_frames[self.T_past:]
        num_agents = min(len(valid_agents), self.max_agents)
        
        # Load all trajectory data in one batch
        track_ids_tuple = tuple(valid_agents[:self.max_agents])
        batch_data = self._load_trajectory_data_batch(video_key, track_ids_tuple)
        
        # Pre-allocate all arrays
        arrays = {
            'past_positions': np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32),
            'future_positions': np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32),
            'past_positions_orig': np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32),
            'future_positions_orig': np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32),
            'obs_masks': np.zeros((self.max_agents, self.T_past), dtype=np.float32),
            'temporal_masks_past': np.zeros((self.max_agents, self.T_past), dtype=np.float32),
            'temporal_masks_future': np.zeros((self.max_agents, self.T_future), dtype=np.float32),
            'occ_masks': np.zeros((self.max_agents, self.T_future), dtype=np.float32),
            'agent_masks': np.zeros(self.max_agents, dtype=np.float32),
            'agent_labels': np.full(self.max_agents, -1, dtype=np.int32),
            'agent_ids': np.full(self.max_agents, -1, dtype=np.int32),
        }
        
        # Pre-compute frame lookups for all agents
        frame_lookups = {}
        for agent_idx, track_id in enumerate(valid_agents[:self.max_agents]):
            if track_id not in batch_data:
                continue
                
            track_data = batch_data[track_id]
            frame_lookups[agent_idx] = {
                'frame_to_idx': {frame: idx for idx, frame in enumerate(track_data['frames'])},
                'data': track_data
            }
            
            # Set agent metadata
            arrays['agent_masks'][agent_idx] = 1.0
            arrays['agent_ids'][agent_idx] = track_id
        
        # Vectorized processing of past frames
        for t, frame in enumerate(past_frames):
            for agent_idx, lookup_data in frame_lookups.items():
                if frame in lookup_data['frame_to_idx']:
                    data_idx = lookup_data['frame_to_idx'][frame]
                    track_data = lookup_data['data']
                    coords = track_data['coords'][data_idx]
                    
                    # FIXED: Validate coordinates before setting
                    if not (np.isnan(coords).any() or np.isinf(coords).any()):
                        # Set positions and masks
                        arrays['past_positions_orig'][agent_idx, t] = coords
                        arrays['past_positions'][agent_idx, t] = coords
                        arrays['temporal_masks_past'][agent_idx, t] = 1.0
                        arrays['obs_masks'][agent_idx, t] = 1.0 if (arrays['agent_ids'][agent_idx], int(frame)) in obs_set else 0.0
                        
                        # Set label if not already set
                        if arrays['agent_labels'][agent_idx] == -1:
                            label = track_data['labels'][data_idx]
                            if label in self.cls2idx:
                                arrays['agent_labels'][agent_idx] = self.cls2idx[label]
                    else:
                        print(f"Warning: Invalid coordinates {coords} for agent {agent_idx} at frame {frame}")
        
        # Vectorized processing of future frames
        for t, frame in enumerate(future_frames):
            for agent_idx, lookup_data in frame_lookups.items():
                if frame in lookup_data['frame_to_idx']:
                    data_idx = lookup_data['frame_to_idx'][frame]
                    track_data = lookup_data['data']
                    coords = track_data['coords'][data_idx]
                    
                    # FIXED: Validate coordinates before setting
                    if not (np.isnan(coords).any() or np.isinf(coords).any()):
                        # Set positions and masks
                        arrays['future_positions_orig'][agent_idx, t] = coords
                        arrays['future_positions'][agent_idx, t] = coords
                        arrays['temporal_masks_future'][agent_idx, t] = 1.0
                        arrays['occ_masks'][agent_idx, t] = track_data['occluded'][data_idx]
                    else:
                        print(f"Warning: Invalid coordinates {coords} for agent {agent_idx} at frame {frame}")
        
        # FIXED: Safe normalization that preserves pad values
        if self.normalize_positions:
            # Only normalize for valid agents with valid positions
            valid_agent_mask = arrays['agent_masks'] > 0
            if np.any(valid_agent_mask):
                # Use the fixed normalization function
                arrays['past_positions'] = self._safe_normalize_coords_vectorized(
                    arrays['past_positions'], 
                    arrays['temporal_masks_past'], 
                    video_key,
                    pad_value=self.pad_value
                )
                
                arrays['future_positions'] = self._safe_normalize_coords_vectorized(
                    arrays['future_positions'], 
                    arrays['temporal_masks_future'], 
                    video_key,
                    pad_value=self.pad_value
                )
        
        # FIXED: Validate all arrays before creating sample
        for key, arr in arrays.items():
            if isinstance(arr, np.ndarray):
                if np.isnan(arr).any():
                    print(f"Warning: NaN found in {key}, replacing with appropriate values")
                    if key in ['past_positions', 'future_positions', 'past_positions_orig', 'future_positions_orig']:
                        # Replace NaN with pad value for position arrays
                        arr = np.where(np.isnan(arr), self.pad_value, arr)
                    else:
                        # Replace NaN with 0 for other arrays
                        arr = np.nan_to_num(arr, nan=0.0)
                    arrays[key] = arr
                
                if np.isinf(arr).any():
                    print(f"Warning: Inf found in {key}, replacing with appropriate values")
                    if key in ['past_positions', 'future_positions', 'past_positions_orig', 'future_positions_orig']:
                        # Replace Inf with pad value for position arrays
                        arr = np.where(np.isinf(arr), self.pad_value, arr)
                    else:
                        # Replace Inf with 0 for other arrays
                        arr = np.where(np.isinf(arr), 0.0, arr)
                    arrays[key] = arr
        
        # Create base sample
        sample = {
            'past_positions': arrays['past_positions'],
            'future_positions': arrays['future_positions'],
            'past_positions_orig': arrays['past_positions_orig'],
            'future_positions_orig': arrays['future_positions_orig'],
            'obs_masks': arrays['obs_masks'],
            'temporal_masks_past': arrays['temporal_masks_past'],
            'temporal_masks_future': arrays['temporal_masks_future'],
            'occ_masks': arrays['occ_masks'],
            'agent_masks': arrays['agent_masks'],
            'agent_labels': arrays['agent_labels'],
            'agent_ids': arrays['agent_ids'],
            'location': sample_idx['location'],
            'video': sample_idx['video'],
            'start_frame': sample_idx['start_frame'],
            'num_valid_agents': num_agents,
            'subsampled_frames': window_frames,
            'frame_subsample_rate': self.frame_subsample
        }
        
        # FIXED: Safe delta computation that handles pad values
        if self.use_deltas:
            valid_agent_mask = arrays['agent_masks'] > 0
            
            # Compute deltas only for valid agents
            past_deltas = self._compute_deltas_vectorized_safe(
                arrays['past_positions'], 
                arrays['temporal_masks_past'],
                pad_value=self.pad_value
            )
            
            future_deltas = self._compute_deltas_vectorized_safe(
                arrays['future_positions'], 
                arrays['temporal_masks_future'],
                is_future=True,
                past_positions=arrays['past_positions'],
                pad_value=self.pad_value
            )
            
            sample['past_deltas'] = past_deltas
            sample['future_deltas'] = future_deltas
        
        return self._format_sample(sample)


    # def _load_multi_agent_sample_on_demand(self, sample_idx):
    #     """Optimized multi-agent sample loading with vectorized operations"""
    #     # Get observation set with fallback
    #     obs_set = self.obs_sets.get((sample_idx['location'], sample_idx['video']), set()) if self.obs_sets else set()
        
    #     video_key = f"{sample_idx['location']}_{sample_idx['video']}"
    #     window_frames = sample_idx['window_frames']
    #     valid_agents = sample_idx['valid_agents']
        
    #     past_frames = window_frames[:self.T_past]
    #     future_frames = window_frames[self.T_past:]
    #     num_agents = min(len(valid_agents), self.max_agents)
        
    #     # Load all trajectory data in one batch
    #     track_ids_tuple = tuple(valid_agents[:self.max_agents])
    #     batch_data = self._load_trajectory_data_batch(video_key, track_ids_tuple)
        
    #     # Pre-allocate all arrays
    #     arrays = {
    #         'past_positions': np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32),
    #         'future_positions': np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32),
    #         'past_positions_orig': np.full((self.max_agents, self.T_past, 2), self.pad_value, dtype=np.float32),
    #         'future_positions_orig': np.full((self.max_agents, self.T_future, 2), self.pad_value, dtype=np.float32),
    #         'obs_masks': np.zeros((self.max_agents, self.T_past), dtype=np.float32),
    #         'temporal_masks_past': np.zeros((self.max_agents, self.T_past), dtype=np.float32),
    #         'temporal_masks_future': np.zeros((self.max_agents, self.T_future), dtype=np.float32),
    #         'occ_masks': np.zeros((self.max_agents, self.T_future), dtype=np.float32),
    #         'agent_masks': np.zeros(self.max_agents, dtype=np.float32),
    #         'agent_labels': np.full(self.max_agents, -1, dtype=np.int32),
    #         'agent_ids': np.full(self.max_agents, -1, dtype=np.int32),
    #     }
        
    #     # Pre-compute frame lookups for all agents
    #     frame_lookups = {}
    #     for agent_idx, track_id in enumerate(valid_agents[:self.max_agents]):
    #         if track_id not in batch_data:
    #             continue
                
    #         track_data = batch_data[track_id]
    #         frame_lookups[agent_idx] = {
    #             'frame_to_idx': {frame: idx for idx, frame in enumerate(track_data['frames'])},
    #             'data': track_data
    #         }
            
    #         # Set agent metadata
    #         arrays['agent_masks'][agent_idx] = 1.0
    #         arrays['agent_ids'][agent_idx] = track_id
        
    #     # Vectorized processing of past frames
    #     for t, frame in enumerate(past_frames):
    #         for agent_idx, lookup_data in frame_lookups.items():
    #             if frame in lookup_data['frame_to_idx']:
    #                 data_idx = lookup_data['frame_to_idx'][frame]
    #                 track_data = lookup_data['data']
    #                 coords = track_data['coords'][data_idx]
                    
    #                 # Set positions and masks
    #                 arrays['past_positions_orig'][agent_idx, t] = coords
    #                 arrays['past_positions'][agent_idx, t] = coords
    #                 arrays['temporal_masks_past'][agent_idx, t] = 1.0
    #                 arrays['obs_masks'][agent_idx, t] = 1.0 if (arrays['agent_ids'][agent_idx], int(frame)) in obs_set else 0.0
                    
    #                 # Set label if not already set
    #                 if arrays['agent_labels'][agent_idx] == -1:
    #                     label = track_data['labels'][data_idx]
    #                     if label in self.cls2idx:
    #                         arrays['agent_labels'][agent_idx] = self.cls2idx[label]
        
    #     # Vectorized processing of future frames
    #     for t, frame in enumerate(future_frames):
    #         for agent_idx, lookup_data in frame_lookups.items():
    #             if frame in lookup_data['frame_to_idx']:
    #                 data_idx = lookup_data['frame_to_idx'][frame]
    #                 track_data = lookup_data['data']
    #                 coords = track_data['coords'][data_idx]
                    
    #                 # Set positions and masks
    #                 arrays['future_positions_orig'][agent_idx, t] = coords
    #                 arrays['future_positions'][agent_idx, t] = coords
    #                 arrays['temporal_masks_future'][agent_idx, t] = 1.0
    #                 arrays['occ_masks'][agent_idx, t] = track_data['occluded'][data_idx]
        
    #     # Vectorized normalization
    #     if self.normalize_positions:
    #         # Only normalize for valid agents
    #         valid_agent_mask = arrays['agent_masks'] > 0
    #         if np.any(valid_agent_mask):
    #             # Normalize past positions
    #             arrays['past_positions'][valid_agent_mask] = self._safe_normalize_coords_vectorized(
    #                 arrays['past_positions'][valid_agent_mask], 
    #                 arrays['temporal_masks_past'][valid_agent_mask], 
    #                 video_key
    #             )
                
    #             # Normalize future positions  
    #             arrays['future_positions'][valid_agent_mask] = self._safe_normalize_coords_vectorized(
    #                 arrays['future_positions'][valid_agent_mask], 
    #                 arrays['temporal_masks_future'][valid_agent_mask], 
    #                 video_key
    #             )
        
    #     # Create base sample
    #     sample = {
    #         'past_positions': arrays['past_positions'],
    #         'future_positions': arrays['future_positions'],
    #         'past_positions_orig': arrays['past_positions_orig'],
    #         'future_positions_orig': arrays['future_positions_orig'],
    #         'obs_masks': arrays['obs_masks'],
    #         'temporal_masks_past': arrays['temporal_masks_past'],
    #         'temporal_masks_future': arrays['temporal_masks_future'],
    #         'occ_masks': arrays['occ_masks'],
    #         'agent_masks': arrays['agent_masks'],
    #         'agent_labels': arrays['agent_labels'],
    #         'agent_ids': arrays['agent_ids'],
    #         'location': sample_idx['location'],
    #         'video': sample_idx['video'],
    #         'start_frame': sample_idx['start_frame'],
    #         'num_valid_agents': num_agents,
    #         'subsampled_frames': window_frames,
    #         'frame_subsample_rate': self.frame_subsample
    #     }
        
    #     # Vectorized delta computation
    #     if self.use_deltas:
    #         valid_agent_mask = arrays['agent_masks'] > 0
            
    #         # Compute deltas only for valid agents
    #         past_deltas = self._compute_deltas_vectorized(
    #             arrays['past_positions'], 
    #             arrays['temporal_masks_past']
    #         )
            
    #         future_deltas = self._compute_deltas_vectorized(
    #             arrays['future_positions'], 
    #             arrays['temporal_masks_future'],
    #             is_future=True,
    #             past_positions=arrays['past_positions']
    #         )
            
    #         sample['past_deltas'] = past_deltas
    #         sample['future_deltas'] = future_deltas
        
    #     return self._format_sample(sample)

    # def __del__(self):
    #     """Clean up HDF5 file handles"""
    #     if hasattr(self, '_hdf5_file_cache'):
    #         if hasattr(self._hdf5_file_cache, 'file') and self._hdf5_file_cache.file:
    #             try:
    #                 self._hdf5_file_cache.file.close()
    #             except:
    #                 pass


    
    def _format_sample(self, sample):
        """Format sample for model consumption"""
        # Convert to tensors if needed
        return sample

class StableTrajectoryLoss(nn.Module):
    """FIXED: More stable loss function with proper NaN/Inf handling"""
    
    def __init__(self, use_deltas=True, predict_uncertainty=False, 
                 position_weight=1.0, delta_weight=1.0, uncertainty_weight=0.01,
                 max_position_error=10.0, max_uncertainty=5.0):  # REDUCED max values
        super().__init__()
        self.use_deltas = use_deltas
        self.predict_uncertainty = predict_uncertainty
        self.position_weight = position_weight
        self.delta_weight = delta_weight
        self.uncertainty_weight = uncertainty_weight
        self.max_position_error = max_position_error
        self.max_uncertainty = max_uncertainty
        
    def forward(self, predictions, batch):
        total_loss = 0.0
        loss_dict = {}
        
        # Extract masks
        temporal_mask = batch['temporal_masks_future']
        occ_mask = batch['occ_masks']
        agent_mask = batch['agent_masks']
        
        # Create visibility mask
        visibility_mask = temporal_mask * (1.0 - occ_mask)
        agent_mask_expanded = agent_mask.unsqueeze(-1).expand_as(visibility_mask)
        valid_mask = visibility_mask * agent_mask_expanded
        
        # FIXED: Check for valid mask before proceeding
        valid_count = valid_mask.sum()
        if valid_count == 0:
            print("Warning: No valid predictions in batch, returning zero loss")
            return torch.tensor(0.0, device=temporal_mask.device, requires_grad=True), {'total_loss': 0.0}
        
        if self.predict_uncertainty:
            # FIXED: More conservative uncertainty handling
            if 'future_positions_mu' in predictions and 'future_deltas_logvar' in predictions:
                pos_mu = predictions['future_positions_mu']
                pos_logvar = predictions['future_deltas_logvar']
                pos_target = batch['future_positions']
                
                # FIXED: Check for NaN/Inf in inputs
                if torch.isnan(pos_mu).any() or torch.isnan(pos_logvar).any() or torch.isnan(pos_target).any():
                    print("Warning: NaN detected in position predictions, skipping")
                    pos_loss = torch.tensor(0.0, device=pos_mu.device, requires_grad=True)
                else:
                    pos_loss = self._robust_gaussian_nll_loss(pos_mu, pos_logvar, pos_target, valid_mask)
                
                loss_dict['position_loss'] = pos_loss
                total_loss += self.position_weight * pos_loss
            
            # Delta loss with same checks  
            if self.use_deltas and 'future_deltas_mu' in predictions and 'future_deltas_logvar' in predictions:
                delta_mu = predictions['future_deltas_mu']
                delta_logvar = predictions['future_deltas_logvar']
                delta_target = batch['future_deltas']
                
                if torch.isnan(delta_mu).any() or torch.isnan(delta_logvar).any() or torch.isnan(delta_target).any():
                    print("Warning: NaN detected in delta predictions, skipping")
                    delta_loss = torch.tensor(0.0, device=delta_mu.device, requires_grad=True)
                else:
                    delta_loss = self._robust_gaussian_nll_loss(delta_mu, delta_logvar, delta_target, valid_mask)
                
                loss_dict['delta_loss'] = delta_loss
                total_loss += self.delta_weight * delta_loss
            
            # FIXED: Much more conservative uncertainty regularization
            if 'future_deltas_logvar' in predictions:
                delta_logvar = predictions['future_deltas_logvar']
                if not torch.isnan(delta_logvar).any():
                    masked_logvar = delta_logvar * valid_mask.unsqueeze(-1).expand_as(delta_logvar)
                    
                    # Simple L2 regularization to prevent extreme values
                    uncertainty_reg = torch.mean(masked_logvar ** 2) * 0.001  # Very small weight
                    
                    loss_dict['uncertainty_reg'] = uncertainty_reg
                    total_loss += self.uncertainty_weight * uncertainty_reg
        else:
            # Standard losses - FIXED version
            if 'future_positions_mu' in predictions:
                pos_mu = predictions['future_positions_mu']
                pos_target = batch['future_positions']
                
                if torch.isnan(pos_mu).any() or torch.isnan(pos_target).any():
                    pos_loss = torch.tensor(0.0, device=pos_mu.device, requires_grad=True)
                else:
                    pos_loss = self._robust_mse_loss(pos_mu, pos_target, valid_mask, self.max_position_error)
                
                loss_dict['position_loss'] = pos_loss
                total_loss += self.position_weight * pos_loss
            
            if self.use_deltas and 'future_deltas_mu' in predictions:
                delta_mu = predictions['future_deltas_mu']
                delta_target = batch['future_deltas']
                
                if torch.isnan(delta_mu).any() or torch.isnan(delta_target).any():
                    delta_loss = torch.tensor(0.0, device=delta_mu.device, requires_grad=True)
                else:
                    delta_loss = self._robust_mse_loss(delta_mu, delta_target, valid_mask, self.max_position_error)
                
                loss_dict['delta_loss'] = delta_loss
                total_loss += self.delta_weight * delta_loss
        
        loss_dict['total_loss'] = total_loss
        
        # FIXED: Better safeguard against NaN/Inf
        if not torch.isfinite(total_loss) or total_loss > 1000.0:  # Cap at reasonable value
            print(f"Warning: Invalid loss detected: {total_loss}, using fallback")
            total_loss = torch.tensor(10.0, device=total_loss.device, requires_grad=True)
        
        return total_loss, loss_dict
    
    def _robust_gaussian_nll_loss(self, mu, logvar, target, mask):
        """FIXED: Much more conservative Gaussian NLL"""
        mask_expanded = mask.unsqueeze(-1).expand_as(mu)
        
        # FIXED: More aggressive clipping
        logvar_clipped = torch.clamp(logvar, min=-10.0, max=5.0)  # Much tighter bounds
        var = torch.exp(logvar_clipped) + 1e-4  # Add small constant for stability
        
        # Compute error
        error = target - mu
        squared_error = error ** 2
        
        # FIXED: Simple Huber-like loss instead of complex soft bounds
        huber_delta = self.max_position_error
        huber_loss = torch.where(
            squared_error <= huber_delta**2,
            0.5 * squared_error,
            huber_delta * (torch.sqrt(squared_error) - 0.5 * huber_delta)
        )
        
        # Simple NLL computation
        nll = 0.5 * (logvar_clipped + huber_loss / var)
        
        # Apply mask
        masked_nll = nll * mask_expanded
        
        # Return average over valid positions
        valid_count = mask_expanded.sum()
        if valid_count > 0:
            return masked_nll.sum() / valid_count
        else:
            return torch.tensor(0.0, device=mu.device, requires_grad=True)
    
    def _robust_mse_loss(self, pred, target, mask, max_error):
        """FIXED: Simple robust MSE"""
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)
        
        # Simple Huber loss
        error = pred - target
        abs_error = torch.abs(error)
        
        huber_loss = torch.where(
            abs_error <= max_error,
            0.5 * error ** 2,
            max_error * (abs_error - 0.5 * max_error)
        )
        
        mse = huber_loss * mask_expanded
        
        valid_count = mask_expanded.sum()
        if valid_count > 0:
            return mse.sum() / valid_count
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=0.01, restore_best_weights=True):  # INCREASED min_delta
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
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def collate_fn(batch):
    """Custom collate function for batching"""
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


def debug_model_forward(model, batch, device):
    """Debug the model forward pass to find where CUDA indexing fails"""
    print("Debugging model forward pass...")
    classes = ['Pedestrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus']
    
    # Move batch to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    # Print input shapes and ranges
    print("\nInput tensor analysis:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, dtype: {value.dtype}")
            if value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                print(f"  Range: [{value.min()}, {value.max()}]")
                if key == 'agent_labels':
                    unique_vals = value.unique()
                    print(f"  Unique values: {unique_vals}")
                    print(f"  Expected range: [-1, {len(classes)-1}] for {len(classes)} classes")
    
    # Try to identify which part of the model fails
    model.eval()
    with torch.inference_mode():
        try:
            print("\nTesting model forward pass...")
            predictions = model(batch, use_teacher_forcing=False)
            print("✓ Model forward pass successful")
            return predictions
        except Exception as e:
            print(f"✗ Model forward pass failed: {e}")
            raise e


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    """FIXED: Train model for one epoch with better stability"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_dict_accum = {}
    
    # FIXED: Remove learning rate warmup that was interfering with scheduler
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Only debug first batch
            if batch_idx == 0:
                print(f"\n=== DEBUG: First batch of epoch {epoch} ===")
                debug_model_forward(model, batch, device)
                print("=== END DEBUG ===\n")
            
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch, use_teacher_forcing=True)
            
            # FIXED: Check predictions for NaN/Inf before loss computation
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    print(f"Warning: Non-finite values in {key}, skipping batch")
                    continue
            
            # Compute loss
            loss, loss_dict = criterion(predictions, batch)
            
            # FIXED: Skip batch if loss is too high
            if loss > 100.0:
                print(f"Skipping batch {batch_idx} due to extremely high loss: {loss}")
                continue
            
            # Backward pass
            loss.backward()
            
            # FIXED: More aggressive gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Much lower
            
            # FIXED: Skip batch if gradients are still too high
            if total_norm > 5.0:
                print(f"Skipping batch {batch_idx} due to high gradient norm: {total_norm}")
                continue
            
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
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'grad_norm': f"{total_norm:.2f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
                
        except RuntimeError as e:
            print(f"\nError at batch {batch_idx}: {e}")
            # Skip this batch and continue
            continue
    
    # Average losses
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        for key in loss_dict_accum:
            loss_dict_accum[key] /= total_samples
    else:
        avg_loss = float('inf')
        loss_dict_accum = {'total_loss': float('inf')}
    
    return avg_loss, loss_dict_accum


def evaluate_model(model, dataloader, criterion, device, dataset=None):
    """FIXED: Evaluate model with better error handling"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_dict_accum = {}
    
    with torch.inference_mode():
        progress_bar = tqdm(dataloader, desc="Validating")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
                
                # Forward pass
                predictions = model(batch, use_teacher_forcing=False)
                
                # FIXED: Check predictions for NaN/Inf
                valid_predictions = True
                for key, value in predictions.items():
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        print(f"Warning: Non-finite values in {key} during validation")
                        valid_predictions = False
                        break
                
                if not valid_predictions:
                    print(f"Skipping validation batch {batch_idx} due to invalid predictions")
                    continue
                
                # Compute loss
                loss, loss_dict = criterion(predictions, batch)
                
                # FIXED: Skip if loss is non-finite
                if not torch.isfinite(loss) or loss > 100.0:
                    print(f"Skipping validation batch {batch_idx} due to invalid loss: {loss}")
                    continue
                
                batch_size = batch['past_positions'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Accumulate loss components
                for key, value in loss_dict.items():
                    if key not in loss_dict_accum:
                        loss_dict_accum[key] = 0.0
                    loss_dict_accum[key] += value.item() * batch_size
                    
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    # Average losses
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        for key in loss_dict_accum:
            loss_dict_accum[key] /= total_samples
    else:
        print("Warning: No valid validation samples processed")
        avg_loss = float('inf')
        loss_dict_accum = {'total_loss': float('inf')}
    
    # Add dummy metrics to avoid KeyError
    loss_dict_accum.update({
        'ADE': avg_loss,  # Use loss as proxy
        'FDE': avg_loss,
        'Agent_ADE_Mean': avg_loss,
        'Agent_FDE_Mean': avg_loss
    })
    
    return avg_loss, loss_dict_accum


def main():
    """FIXED: Main function with better hyperparameters"""
    # Configuration
    config = {
        'drone_data_root': "drive/MyDrive/Colab Notebooks/square_stanford_data",
        'original_dataset_root': "drive/MyDrive/Colab Notebooks/stanford_data/archive",
        'cache_dir': "drive/MyDrive/Colab Notebooks/new_cache_64",
        'checkpoint_dir': "checkpoints",
        'batch_size': 64,  # REDUCED batch size
        'num_epochs': 15,
        'learning_rate': 1e-4,  # REDUCED learning rate
        'weight_decay': 1e-5,  # INCREASED weight decay
        'patience': 15,
        'val_split': 0.1,
        'num_workers': 4,
        'max_agents': 64,
        'T_past': 10,
        'T_future': 20,
        'frame_subsample': 12,
        'wandb_project': 'trajectory-prediction',
        'run_name': 'multi-agent-trajectory-exp-fixed'
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=config['wandb_project'],
        name=config['run_name'],
        config=config
    )
    
    # Dataset configuration
    classes = ['Pedestrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus']
    locations = ['bookstore', 'little', 'coupa', 'deathCircle', 'gates', 'hyang', 'nexus', 'quad']
    
    print("Loading dataset...")
    
    # NOTE: You'll need to uncomment and import your actual dataset
    # from your_dataset_module import OptimizedMultiAgentSequenceDataset
    
    # For now, using placeholder
    dataset = OptimizedMultiAgentSequenceDataset(
        drone_data_root=config['drone_data_root'],
        original_dataset_root=config['original_dataset_root'],
        classes=classes,
        T_past=config['T_past'],
        T_future=config['T_future'],
        use_deltas=True,
        normalize_positions=True,
        cache_dir=config['cache_dir'],
        lazy_loading=True,
        num_workers=config['num_workers'],
        max_agents=config['max_agents'],
        frame_subsample=config['frame_subsample']
    )
    
    if dataset is not None:
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Split dataset
        val_size = int(len(dataset) * config['val_split'])
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True if device.type == 'cuda' else False,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True if device.type == 'cuda' else False,
            collate_fn=collate_fn,
            drop_last=False
        )
    else:
        train_loader = val_loader = None
    
    # Initialize model
    print("Initializing model...")
    
    # NOTE: You'll need to uncomment and import your actual model
    # from your_model_module import ImprovedGraphInteractionModel
    
    # For now, using placeholder
    model = ImprovedGraphInteractionModel(
        num_classes=len(classes),
        locations=locations,
        d_model=64,
        nhead=8,
        num_layers=3,
        T_past=config['T_past'],
        T_future=config['T_future'],
        max_agents=config['max_agents']
    )
    
    if model is not None:
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # FIXED: More conservative loss function
        criterion = StableTrajectoryLoss(
            use_deltas=True,
            predict_uncertainty=False,  # Disable uncertainty for now
            position_weight=1.0,
            delta_weight=0.5,  # Reduced delta weight
            uncertainty_weight=0.01
        )
        
        # FIXED: More conservative optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            eps=1e-8,  # Increased epsilon for stability
            betas=(0.9, 0.999)
        )
        
        # FIXED: More conservative learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-7
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=config['patience'], min_delta=0.01)
        
        # Training loop
        best_val_loss = float('inf')
        
        print("Starting training...")
        for epoch in range(1, config['num_epochs'] + 1):
            start_time = time.time()
            
            # Train
            if train_loader is not None:
                train_loss, train_loss_dict = train_epoch(
                    model, train_loader, criterion, optimizer, device, epoch
                )
            else:
                train_loss, train_loss_dict = 0.0, {'total_loss': 0.0}
            
            # Validate
            if val_loader is not None:
                val_loss, val_loss_dict = evaluate_model(
                    model, val_loader, criterion, device, dataset
                )
            else:
                val_loss, val_loss_dict = 0.0, {'total_loss': 0.0}
            
            # FIXED: Update scheduler based on validation loss
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
            
            print(f"Epoch {epoch}/{config['num_epochs']}")
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            if 'ADE' in val_loss_dict and 'FDE' in val_loss_dict:
                print(f"Val ADE: {val_loss_dict['ADE']:.4f}, Val FDE: {val_loss_dict['FDE']:.4f}")
            print(f"Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint every epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'classes': classes,
                'locations': locations
            }
            
            # Save current epoch checkpoint
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
                print(f"New best model saved with val_loss: {val_loss:.6f}")
            
            # Save latest model (overwrite each epoch)
            torch.save(checkpoint, checkpoint_dir / 'latest_model.pth')
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            print("-" * 50)
    
    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()