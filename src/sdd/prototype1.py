import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os 


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, max_len, 2).float() * (-math.log(10000)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SpatialEncoding(nn.Module):
    def __init__(self, d_model, max_pos=10000):
        super(SpatialEncoding, self).__init__()
        self.d_model = d_model
        self.max_pos = max_pos
        self.pos_embed = nn.Linear(2, d_model)
        self.half_dim = d_model//2

        div_term = torch.exp(
            torch.arange(0, self.half_dim, 2).float() *
            ( -math.log(10000.0) / self.half_dim )
        )
        self.register_buffer('div_term', div_term)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: (B, T, 2)  — normalized x,y in [0,1]
        returns:    (B, T, d_model)
        """
        B, T, _ = positions.shape
        scaled = positions * self.max_pos   # now in [0, max_pos]

        pe_x = torch.zeros(B, T, self.half_dim, device=positions.device)

        pe_x[:, :, 0::2] = torch.sin(scaled[:, :, 0:1] * self.div_term)
        pe_x[:, :, 1::2] = torch.cos(scaled[:, :, 0:1] * self.div_term)

        pe_y = torch.zeros(B, T, self.half_dim, device=positions.device)
        pe_y[:, :, 0::2] = torch.sin(scaled[:, :, 1:2] * self.div_term)
        pe_y[:, :, 1::2] = torch.cos(scaled[:, :, 1:2] * self.div_term)
        
        # concatenate x and y → (B, T, d_model)
        return torch.cat([pe_x, pe_y], dim=-1)
    

class AgentSequenceDataset(Dataset):
    def __init__(self, drone_data_root, original_dataset_root, classes,
                 T_past=10, T_future=10, use_deltas=True, normalize_positions=True):
        self.drone_data_root = drone_data_root
        self.orig_root = original_dataset_root
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        self.T_past = T_past
        self.T_future = T_future
        self.use_deltas = use_deltas
        self.normalize_positions = normalize_positions
        self.samples = []
        
        # For video-wise normalization - compute per-video statistics
        self.video_stats = self._compute_video_stats() if normalize_positions else None

        # Precompute obs sets per (loc,vid): which (trackId,frame) were seen
        obs = {}  # obs[(loc,vid)] = set of (trackId,frame)
        for loc in os.listdir(drone_data_root):
            loc_path = os.path.join(drone_data_root, loc)
            if not os.path.isdir(loc_path): continue
            if loc == 'hyang': continue
            for dr in os.listdir(loc_path):
                dr_path = os.path.join(loc_path, dr)
                if not os.path.isdir(dr_path): continue
                for fname in os.listdir(dr_path):
                    if not fname.endswith("_annotations.txt"): continue
                    vid = fname.replace("_annotations.txt","")
                    key = (loc, vid)
                    obs.setdefault(key, set())
                    df = pd.read_csv(os.path.join(dr_path,fname), sep=' ', header=None,
                                     names=['trackId','xmin','ymin','xmax','ymax',
                                            'frame','lost','occluded','generated','label'])
                    for tid,fr in zip(df['trackId'], df['frame']):
                        obs[key].add((tid, int(fr)))

        # Build samples using original dataset for full tracks
        for loc in os.listdir(original_dataset_root + "/annotations"):
            if loc == 'hyang' or loc == ".DS_Store": continue
            loc_path = os.path.join(original_dataset_root, "annotations", loc)
            if not os.path.isdir(loc_path):
                continue
            for vid in os.listdir(loc_path):
                # load original annotations
                orig_file = os.path.join(original_dataset_root,"annotations",loc,vid,"annotations.txt")
                if not os.path.isfile(orig_file): continue
                odf = pd.read_csv(orig_file, sep=' ', header=None,
                                  names=['trackId','xmin','ymin','xmax','ymax',
                                         'frame','lost','occluded','generated','label'])
                # compute centers
                odf['x'] = (odf.xmin + odf.xmax)/2
                odf['y'] = (odf.ymin + odf.ymax)/2
                # keep only labels in classes
                odf = odf[odf['label'].isin(self.classes)]
                # group by track
                key = (loc, vid)
                seen = obs.get(key, set())
                video_key = f"{loc}_{vid}"
                
                for tid, grp in odf.groupby('trackId'):
                    grp = grp.sort_values('frame')
                    frames = grp['frame'].values
                    coords = np.stack([grp['x'].values, grp['y'].values], axis=1)
                    occs   = (grp['occluded']==0).astype(np.float32).values
                    # build observation mask
                    obs_mask = np.array([1.0 if (tid, int(f)) in seen else 0.0 for f in frames],
                                        dtype=np.float32)
                    L = len(frames)
                    window = self.T_past + self.T_future
                    for i in range(L - window + 1):
                        # ensure continuous frames in original
                        if frames[i+self.T_past-1] - frames[i] != self.T_past-1:
                            continue
                        if frames[i+window-1]   - frames[i+self.T_past] != self.T_future-1:
                            continue
                        # require at least one observed in past
                        if obs_mask[i:i+self.T_past].sum() < 1:
                            continue
                        
                        past_coords = coords[i:i+self.T_past]
                        future_coords = coords[i+self.T_past:i+window]
                        past_obs = obs_mask[i:i+self.T_past]
                        future_occ = occs[i+self.T_past:i+window]
                        label = grp['label'].values[i+self.T_past-1]
                        idx = self.cls2idx[label]
                        
                        # Store original coordinates for spatial encoding
                        orig_past_coords = past_coords.copy()
                        orig_future_coords = future_coords.copy()
                        
                        # Normalize positions if requested (video-wise normalization)
                        if self.normalize_positions:
                            past_coords = self._normalize_coords(past_coords, video_key)
                            future_coords = self._normalize_coords(future_coords, video_key)
                        
                        # Create sample dictionary
                        sample = {
                            'past_positions': past_coords.astype(np.float32),
                            'future_positions': future_coords.astype(np.float32),
                            'past_positions_orig': orig_past_coords.astype(np.float32),  # For spatial encoding
                            'future_positions_orig': orig_future_coords.astype(np.float32),  # For spatial encoding
                            'obs_mask': past_obs,
                            'occ_mask': future_occ.astype(np.float32),
                            'label': idx,
                            'location': loc,
                            'video': vid,
                            'track_id': tid,
                            'start_frame': frames[i]
                        }
                        
                        # Add delta features if requested
                        if self.use_deltas:
                            # Past deltas (movement between consecutive frames)
                            past_deltas = np.diff(past_coords, axis=0)  # (T_past-1, 2)
                            # Pad with zeros for first frame (no previous delta)
                            past_deltas = np.concatenate([np.zeros((1, 2)), past_deltas], axis=0)
                            
                            # Future deltas
                            future_deltas = np.diff(future_coords, axis=0)  # (T_future-1, 2)
                            # Add the delta from last past to first future
                            transition_delta = future_coords[0] - past_coords[-1]
                            future_deltas = np.concatenate([transition_delta.reshape(1, 2), future_deltas], axis=0)
                            
                            sample['past_deltas'] = past_deltas.astype(np.float32)
                            sample['future_deltas'] = future_deltas.astype(np.float32)
                        
                        self.samples.append(sample)

    def _compute_video_stats(self):
        """Compute per-video position statistics for normalization"""
        video_stats = {}
        
        for loc in os.listdir(self.orig_root + "/annotations"):
            if loc == 'hyang' or loc == ".DS_Store": continue
            loc_path = os.path.join(self.orig_root, "annotations", loc)
            if not os.path.isdir(loc_path):
                continue
            for vid in os.listdir(loc_path):
                orig_file = os.path.join(self.orig_root,"annotations",loc,vid,"annotations.txt")
                if not os.path.isfile(orig_file): continue
                
                video_key = f"{loc}_{vid}"
                odf = pd.read_csv(orig_file, sep=' ', header=None,
                                  names=['trackId','xmin','ymin','xmax','ymax',
                                         'frame','lost','occluded','generated','label'])
                odf['x'] = (odf.xmin + odf.xmax)/2
                odf['y'] = (odf.ymin + odf.ymax)/2
                
                coords = odf[['x', 'y']].values
                if len(coords) > 0:
                    video_stats[video_key] = {
                        'mean': np.mean(coords, axis=0),
                        'std': np.std(coords, axis=0),
                        'min': np.min(coords, axis=0),
                        'max': np.max(coords, axis=0)
                    }
        
        return video_stats

    def _normalize_coords(self, coords, video_key):
        """Normalize coordinates using video-specific statistics"""
        if self.video_stats is None or video_key not in self.video_stats:
            return coords
        
        stats = self.video_stats[video_key]
        # Use standardization (zero mean, unit variance)
        normalized = (coords - stats['mean']) / (stats['std'] + 1e-8)
        return normalized

    def get_video_stats(self, video_key):
        """Return video-specific statistics for denormalization"""
        if self.video_stats is None:
            return None
        return self.video_stats.get(video_key, None)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        result = {
            'past_positions': torch.from_numpy(s['past_positions']),     # (T_past, 2) - normalized
            'future_positions': torch.from_numpy(s['future_positions']), # (T_future, 2) - normalized
            'past_positions_orig': torch.from_numpy(s['past_positions_orig']),  # (T_past, 2) - original scale
            'future_positions_orig': torch.from_numpy(s['future_positions_orig']),  # (T_future, 2) - original scale
            'obs_mask': torch.from_numpy(s['obs_mask']),                 # (T_past,)
            'occ_mask': torch.from_numpy(s['occ_mask']),                 # (T_future,)
            'label': torch.tensor(s['label']),
            'location': s['location'],
            'video': s['video'],
            'track_id': s['track_id'],
            'start_frame': s['start_frame']
        }
        
        if self.use_deltas:
            result['past_deltas'] = torch.from_numpy(s['past_deltas'])     # (T_past, 2)
            result['future_deltas'] = torch.from_numpy(s['future_deltas']) # (T_future, 2)
        
        return result


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

def deltas_to_positions(start_pos, deltas):
    """Convert delta predictions back to absolute positions"""
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



class LandscapeAwareTrajectoryPredictor(nn.Module):
    def __init__(self, num_classes, locations, d_model=256, nhead=8, num_layers=6, 
                 T_past=10, T_future=10, use_deltas=True):
        super(LandscapeAwareTrajectoryPredictor, self).__init__()
        
        self.d_model = d_model
        self.T_past = T_past
        self.T_future = T_future
        self.use_deltas = use_deltas
        self.num_classes = num_classes

        self.location_embedding = nn.Embedding(len(locations), d_model)
        self.location_to_idx = {loc:i for i,loc in enumerate(locations)}

        self.class_embedding = nn.Embedding(num_classes, d_model)
        
        # Spatial encoding for absolute positions (uses original coordinates)
        self.spatial_encoding = SpatialEncoding(d_model)
        
        # Temporal encoding
        self.temporal_encoding = PositionalEncoding(d_model)

        self.pos_projection = nn.Linear(2, d_model)
        if use_deltas:
            self.delta_projection = nn.Linear(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, 
                                                  dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Transformer decoder for future prediction
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model*4,
                                                  dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        if use_deltas:
            self.delta_head = nn.Linear(d_model, 2)
        self.position_head = nn.Linear(d_model, 2)

        
    def forward(self, batch):
        batch_size = batch['past_positions'].size(0)
        device = batch['past_positions'].device

        location_indices = torch.tensor([self.location_to_idx(loc) for loc in batch['location']], device = device)
        location_emb = self.location_embedding(location_indices) 

        class_emb = torch.tensor([self.class_embedding(i) for i in batch['label']])

        past_positions = batch['past_positions']  # (batch, T_past, 2) - normalized
        past_positions_orig = batch['past_positions_orig']  # (batch, T_past, 2) - original scale
        obs_mask = batch['obs_mask']  # (batch, T_past)

        pos_emb = self.pos_projection(past_positions)  # (batch, T_past, d_model)
        
        # Spatial encoding (using original coordinates for landscape learning)
        spatial_emb = self.spatial_encoding(past_positions_orig)  # (batch, T_past, d_model)
        
        # Combine embeddings
        past_emb = pos_emb + spatial_emb

        # Add location and class context
        past_emb = past_emb + location_emb.unsqueeze(1) + class_emb.unsqueeze(1)

        past_emb = self.temporal_encoding(past_emb.transpose(0, 1)).transpose(0, 1)
        
        # Create attention mask for observed positions
        attention_mask = (obs_mask == 0)  # True for unobserved positions
        
        # Encode past sequence
        memory = self.transformer_encoder(past_emb, src_key_padding_mask=attention_mask)
        
        # Prepare decoder input (start with last observed position)
        # Find last observed position for each sequence
        last_obs_idx = []
        for i in range(batch_size):
            obs_indices = torch.where(obs_mask[i] == 1)[0]
            if len(obs_indices) > 0:
                last_obs_idx.append(obs_indices[-1].item())
            else:
                last_obs_idx.append(0)  # fallback

        # Initialize decoder input with last observed position
        decoder_input = torch.zeros(batch_size, self.T_future, self.d_model, device=device)
        for i, idx in enumerate(last_obs_idx):
            decoder_input[i, 0] = memory[i, idx]
        
        # For decoder, we also need to consider future spatial context
        # Use the last observed position as starting point for spatial encoding
        if hasattr(batch, 'future_positions_orig'):
            # If we have future original positions (for training), use them for spatial encoding
            future_spatial_emb = self.spatial_encoding(batch['future_positions_orig'])
            decoder_input = decoder_input + future_spatial_emb

        future_emb = self.transformer_decoder(decoder_input, memory, 
                                            memory_key_padding_mask=attention_mask)
        
        # Output predictions
        outputs = {}
        
        if self.use_deltas:
            delta_pred = self.delta_head(future_emb)  # (batch, T_future, 2)
            outputs['future_deltas'] = delta_pred
            
            # Convert deltas to positions
            start_positions = torch.stack([past_positions[i, last_obs_idx[i]] 
                                         for i in range(batch_size)])
            outputs['future_positions'] = self.deltas_to_positions(start_positions, delta_pred)
        else:
            outputs['future_positions'] = self.position_head(future_emb)
        
        return outputs
    
    def deltas_to_positions(self, start_pos, deltas):
        """Convert delta predictions back to absolute positions"""
        batch_size, seq_len, _ = deltas.shape
        positions = torch.zeros_like(deltas)
        
        # First position is start_pos + first delta
        positions[:, 0] = start_pos + deltas[:, 0]
        
        # Subsequent positions are cumulative
        for i in range(1, seq_len):
            positions[:, i] = positions[:, i-1] + deltas[:, i]
            
        return positions




































































































































































class DroneSimulator:
    def __init__(self, dataset_path: str, output_path: str = "square_stanford_data"):
        """
        Args:
            dataset_path: Path to the Stanford drone dataset
            output_path: Path where the new multi-drone dataset will be created
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.locations = self._get_locations()
        self.default_fov_side = 420  # Default FOV side length in pixels (2 * previous radius)
        
    def _get_locations(self) -> List[str]:
        """Get all available locations from the dataset"""
        annotations_path = self.dataset_path / "annotations"
        locations = []
        
        if annotations_path.exists():
            for item in annotations_path.iterdir():
                if item.is_dir():
                    locations.append(item.name)
        
        return locations
    
    def _load_annotations(self, location: str) -> Dict[str, pd.DataFrame]:
        annotations = {}
        location_path = self.dataset_path / "annotations" / location
        
        if not location_path.exists():
            print(f"Location {location} not found in dataset")
            return annotations
        
        for video_dir in location_path.iterdir():
            if video_dir.is_dir():
                annotation_file = video_dir / "annotations.txt"
                if annotation_file.exists():
                    try:
                        # Load annotations - assuming space-separated format
                        # Common format: trackId xmin ymin xmax ymax frame lost occluded generated label
                        df = pd.read_csv(annotation_file, sep='\s+', header=None,
                                       names=['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 
                                             'frame', 'lost', 'occluded', 'generated', 'label'])
                        annotations[video_dir.name] = df
                    except Exception as e:
                        print(f"Error loading annotations for {video_dir.name}: {e}")
        
        return annotations
    
    def _load_reference_image(self, location: str, video: str) -> np.ndarray:
        """
        Load reference image for a location/video
        
        Args:
            location: Location name
            video: Video name
            
        Returns:
            Reference image as numpy array
        """
        ref_path = self.dataset_path / "annotations" / location / video / "reference.jpg"
        if ref_path.exists():
            return cv2.imread(str(ref_path))
        return None
    
    def _transform_to_local_coordinates(self, annotations: pd.DataFrame, 
                                      drone_pos: Tuple[float, float],
                                      fov_side: float) -> pd.DataFrame:
        """
        Transform annotations to local drone coordinates and filter by square FOV
        
        Args:
            annotations: Original annotations DataFrame
            drone_pos: (x, y) position of the drone
            fov_side: FOV side length
            
        Returns:
            Filtered and transformed annotations DataFrame
        """
        if annotations.empty:
            return annotations
        
        # Calculate center points of bounding boxes
        annotations = annotations.copy()
        annotations['center_x'] = (annotations['xmin'] + annotations['xmax']) / 2
        annotations['center_y'] = (annotations['ymin'] + annotations['ymax']) / 2
        
        # Filter objects within square FOV
        half_side = fov_side / 2
        x_in_fov = (annotations['center_x'] >= drone_pos[0] - half_side) & \
                   (annotations['center_x'] <= drone_pos[0] + half_side)
        y_in_fov = (annotations['center_y'] >= drone_pos[1] - half_side) & \
                   (annotations['center_y'] <= drone_pos[1] + half_side)
        
        in_fov = x_in_fov & y_in_fov
        filtered_annotations = annotations[in_fov].copy()
        
        if filtered_annotations.empty:
            return filtered_annotations
        
        # Transform to local coordinates (drone at origin)
        filtered_annotations['local_xmin'] = filtered_annotations['xmin'] - drone_pos[0]
        filtered_annotations['local_ymin'] = filtered_annotations['ymin'] - drone_pos[1]
        filtered_annotations['local_xmax'] = filtered_annotations['xmax'] - drone_pos[0]
        filtered_annotations['local_ymax'] = filtered_annotations['ymax'] - drone_pos[1]
        filtered_annotations['local_center_x'] = filtered_annotations['center_x'] - drone_pos[0]
        filtered_annotations['local_center_y'] = filtered_annotations['center_y'] - drone_pos[1]
        
        return filtered_annotations
    
    
    def _save_drone_data(self, location: str, drone_id: str, video: str,
                        local_annotations: pd.DataFrame, fov_image: np.ndarray, x, y):
        """
        Save drone-specific data to the output directory
        
        Args:
            location: Location name
            drone_id: Drone identifier
            video: Video name
            local_annotations: Transformed annotations
            fov_image: FOV image
        """
        # Create directory structure
        drone_dir = self.output_path / location / drone_id
        drone_dir.mkdir(parents=True, exist_ok=True)
        
        # Save annotations
        if not local_annotations.empty:
            annotation_file = drone_dir / f"{video}_annotations.txt"
            # Save with local coordinates
            columns_to_save = ['trackId', 'local_xmin', 'local_ymin', 'local_xmax', 'local_ymax',
                             'frame', 'lost', 'occluded', 'generated', 'label']
            local_annotations[columns_to_save].to_csv(annotation_file, sep=' ', 
                                                    header=False, index=False)
        
        # Save FOV image
        if fov_image is not None and fov_image.size != 0:
            fov_file = drone_dir / f"{video}_fov.jpg"
            cv2.imwrite(str(fov_file), fov_image)
        
        # Save metadata
        metadata = {
            'location': location,
            'drone_id': drone_id,
            'video': video,
            'total_annotations': len(local_annotations),
            'fov_image_shape': fov_image.shape if fov_image is not None else None,
            'fov_type': 'square',
            'x_position': x,
            'y_position': y
        }
        
        metadata_file = drone_dir / f"{video}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def simulate_drones(self, drone_positions: Dict[str, List[Tuple[float, float]]], 
                       fov_side: float = None):
        """
        Simulate multiple drones for each location with square FOVs
        
        Args:
            drone_positions: Dictionary mapping location names to lists of (x, y) positions
            fov_side: FOV side length for all drones (uses default if None)
        """
        if fov_side is None:
            fov_side = self.default_fov_side
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        for location, positions in drone_positions.items():
            if location not in self.locations:
                print(f"Warning: Location '{location}' not found in dataset")
                continue
            
            # Validate drone positions
            if not self._validate_drone_positions(positions, fov_side):
                print(f"Warning: Some drone square FOVs overlap in location '{location}'")
                continue
            
            print(f"Processing location: {location}")
            
            # Load annotations for this location
            annotations = self._load_annotations(location)
            
            if not annotations:
                print(f"No annotations found for location: {location}")
                continue
            
            # Process each drone
            for drone_idx, drone_pos in enumerate(positions):
                drone_id = f"drone{drone_idx + 1}"
                print(f"  Processing {drone_id} at position {drone_pos}")
                
                # Process each video for this drone
                for video_name, video_annotations in annotations.items():
                    # Load reference image
                    reference_image = self._load_reference_image(location, video_name)
                    
                    # Transform annotations to local coordinates
                    local_annotations = self._transform_to_local_coordinates(
                        video_annotations, drone_pos, fov_side)
                    
                    # Create square FOV image
                    fov_image = self._create_fov_image(reference_image, drone_pos, fov_side)
                    
                    # Save drone data
                    self._save_drone_data(location, drone_id, video_name, 
                                        local_annotations, fov_image, drone_pos[0], drone_pos[1])
        
        print("Drone simulation completed!")
        print(f"Output saved to: {self.output_path}")




class AgentSequenceDataset(Dataset):
    def __init__(self, drone_data_root, original_dataset_root, classes,
                 T_past=10, T_future=10):
        self.drone_data_root = drone_data_root
        self.orig_root = original_dataset_root
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        self.T_past = T_past
        self.T_future = T_future
        self.samples = []

        # Precompute obs sets per (loc,vid): which (trackId,frame) were seen
        obs = {}  # obs[(loc,vid)] = set of (trackId,frame)
        for loc in os.listdir(drone_data_root):
            loc_path = os.path.join(drone_data_root, loc)
            if not os.path.isdir(loc_path): continue
            if loc == 'hyang': continue
            for dr in os.listdir(loc_path):
                dr_path = os.path.join(loc_path, dr)
                if not os.path.isdir(dr_path): continue
                # metadata gives nothing for obs
                for fname in os.listdir(dr_path):
                    if not fname.endswith("_annotations.txt"): continue
                    vid = fname.replace("_annotations.txt","")
                    key = (loc, vid)
                    obs.setdefault(key, set())
                    df = pd.read_csv(os.path.join(dr_path,fname), sep=' ', header=None,
                                     names=['trackId','xmin','ymin','xmax','ymax',
                                            'frame','lost','occluded','generated','label'])
                    for tid,fr in zip(df['trackId'], df['frame']):
                        obs[key].add((tid, int(fr)))

        # Build samples using original dataset for full tracks
        for loc in os.listdir(original_dataset_root + "/annotations"):
            if loc == 'hyang' or loc == ".DS_Store": continue
            if not os.path.isdir(loc_path):            # ← skip files like .DS_Store
                continue
            for vid in os.listdir(os.path.join(original_dataset_root,"annotations",loc)):
                # load original annotations
                orig_file = os.path.join(original_dataset_root,"annotations",loc,vid,"annotations.txt")
                if not os.path.isfile(orig_file): continue
                odf = pd.read_csv(orig_file, sep=' ', header=None,
                                  names=['trackId','xmin','ymin','xmax','ymax',
                                         'frame','lost','occluded','generated','label'])
                # compute centers
                odf['x'] = (odf.xmin + odf.xmax)/2
                odf['y'] = (odf.ymin + odf.ymax)/2
                # keep only labels in classes
                odf = odf[odf['label'].isin(self.classes)]
                # group by track
                key = (loc, vid)
                seen = obs.get(key, set())
                for tid, grp in odf.groupby('trackId'):
                    grp = grp.sort_values('frame')
                    frames = grp['frame'].values
                    coords = np.stack([grp['x'].values, grp['y'].values], axis=1)
                    occs   = (grp['occluded']==0).astype(np.float32).values
                    # build observation mask
                    obs_mask = np.array([1.0 if (tid, int(f)) in seen else 0.0 for f in frames],
                                        dtype=np.float32)
                    L = len(frames)
                    window = self.T_past + self.T_future
                    for i in range(L - window + 1):
                        # ensure continuous frames in original
                        if frames[i+self.T_past-1] - frames[i] != self.T_past-1:
                            continue
                        if frames[i+window-1]   - frames[i+self.T_past] != self.T_future-1:
                            continue
                        # require at least one observed in past
                        if obs_mask[i:i+self.T_past].sum() < 1:
                            continue
                        past   = coords[i:i+self.T_past]
                        future = coords[i+self.T_past:i+window]
                        past_obs = obs_mask[i:i+self.T_past]
                        future_occ = occs[i+self.T_past:i+window]
                        label = grp['label'].values[i+self.T_past-1]
                        idx = self.cls2idx[label]
                        self.samples.append({
                            'past': past.astype(np.float32),
                            'obs_mask': past_obs,
                            'future': future.astype(np.float32),
                            'occ_mask': future_occ.astype(np.float32),
                            'label': idx
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'past':     torch.from_numpy(s['past']),     # (T_past,2)
            'obs_mask': torch.from_numpy(s['obs_mask']), # (T_past,)
            'future':   torch.from_numpy(s['future']),   # (T_future,2)
            'occ_mask': torch.from_numpy(s['occ_mask']), # (T_future,)
            'label':    torch.tensor(s['label'])
        }