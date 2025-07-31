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
        
    def _compute_deltas_vectorized(self, positions, temporal_masks, is_future=False, past_positions=None):
        """Vectorized delta computation"""
        max_agents, seq_len, _ = positions.shape
        deltas = np.zeros_like(positions)
        
        for agent_idx in range(max_agents):
            if not np.any(temporal_masks[agent_idx]):
                continue
                
            valid_indices = np.where(temporal_masks[agent_idx] > 0)[0]
            if len(valid_indices) <= 1 and not is_future:
                continue
            
            if is_future and past_positions is not None:
                # Handle transition from past to future
                past_mask = temporal_masks[agent_idx] if past_positions.shape[1] == seq_len else None
                if past_mask is not None:
                    past_valid = np.where(past_mask > 0)[0]
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
    def _normalize_coords_vectorized(self, coords_array, temporal_mask, video_key):
        """Vectorized coordinate normalization"""
        if self.video_stats is None or video_key not in self.video_stats:
            return coords_array
        
        stats = self.video_stats[video_key]
        mean = stats['mean'].reshape(1, 1, 2)  # Broadcast shape
        std = stats['std'].reshape(1, 1, 2) + 1e-8
        
        # Only normalize where temporal_mask is True
        normalized = coords_array.copy()
        mask_expanded = temporal_mask[:, :, np.newaxis]  # Shape: (agents, time, 1)
        normalized = np.where(mask_expanded, (coords_array - mean) / std, coords_array)
        
        return normalized
        
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
        """Optimized multi-agent sample loading with vectorized operations"""
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
        
        # Vectorized processing of future frames
        for t, frame in enumerate(future_frames):
            for agent_idx, lookup_data in frame_lookups.items():
                if frame in lookup_data['frame_to_idx']:
                    data_idx = lookup_data['frame_to_idx'][frame]
                    track_data = lookup_data['data']
                    coords = track_data['coords'][data_idx]
                    
                    # Set positions and masks
                    arrays['future_positions_orig'][agent_idx, t] = coords
                    arrays['future_positions'][agent_idx, t] = coords
                    arrays['temporal_masks_future'][agent_idx, t] = 1.0
                    arrays['occ_masks'][agent_idx, t] = track_data['occluded'][data_idx]
        
        # Vectorized normalization
        if self.normalize_positions:
            # Only normalize for valid agents
            valid_agent_mask = arrays['agent_masks'] > 0
            if np.any(valid_agent_mask):
                # Normalize past positions
                arrays['past_positions'][valid_agent_mask] = self._normalize_coords_vectorized(
                    arrays['past_positions'][valid_agent_mask], 
                    arrays['temporal_masks_past'][valid_agent_mask], 
                    video_key
                )
                
                # Normalize future positions  
                arrays['future_positions'][valid_agent_mask] = self._normalize_coords_vectorized(
                    arrays['future_positions'][valid_agent_mask], 
                    arrays['temporal_masks_future'][valid_agent_mask], 
                    video_key
                )
        
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
        
        # Vectorized delta computation
        if self.use_deltas:
            valid_agent_mask = arrays['agent_masks'] > 0
            
            # Compute deltas only for valid agents
            past_deltas = self._compute_deltas_vectorized(
                arrays['past_positions'], 
                arrays['temporal_masks_past']
            )
            
            future_deltas = self._compute_deltas_vectorized(
                arrays['future_positions'], 
                arrays['temporal_masks_future'],
                is_future=True,
                past_positions=arrays['past_positions']
            )
            
            sample['past_deltas'] = past_deltas
            sample['future_deltas'] = future_deltas
        
        return self._format_sample(sample)

    def __del__(self):
        """Clean up HDF5 file handles"""
        if hasattr(self, '_hdf5_file_cache'):
            if hasattr(self._hdf5_file_cache, 'file') and self._hdf5_file_cache.file:
                try:
                    self._hdf5_file_cache.file.close()
                except:
                    pass
    
    def _format_sample(self, sample):
        """Format sample for model consumption"""
        # Convert to tensors if needed
        return sample