import os
import numpy as np
import pandas as pd
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

# class AgentSequenceDataset(Dataset):
#     """
#     PyTorch Dataset yielding per-agent trajectory sequences.

#     For each agent track, returns:
#       - past:  Tensor[T_past, 2]  (x,y) positions
#       - future:Tensor[T_future, 2] (x,y) positions
#       - label: Tensor               class index
#     Only sequences where both past and future windows are fully observed are included.
#     """
#     def __init__(self,
#                  drone_data_root: str,
#                  original_dataset_root: str,
#                  classes: list,
#                  T_past:int = 10,
#                  T_future:int = 10):
#         self.drone_data_root = drone_data_root
#         self.orig_root = original_dataset_root
#         self.classes = classes
#         self.cls2idx = {c: i for i, c in enumerate(classes)}
#         self.T_past = T_past
#         self.T_future = T_future
#         self.samples = [] 
#         drone_centres = {}
#         for loc in os.listdir(self.drone_data_root):
#             loc_path = os.path.join(self.drone_data_root, loc)
#             for dr in loc_path:
#                 js = json.load(loc_path, dr, f"video0_metadata.json")
#                 drone_centres[(loc, dr)] = (js["x_position"], js["y_position"])

#         for loc in os.listdir(self.drone_data_root):
#             loc_path = os.path.join(self.drone_data_root, loc)
#             drone1_dir = os.path.join(loc_path, "drone1")
#             vids = [filen.replace("_annotations.txt", "") for filen in os.listdir(drone1_dir) if filen.endswith("_annotations.txt")]
#             for vid in vids:
#                 dfs = []
#                 for dr in os.listdir(loc_path):
#                     #js = json.load(os.path.join(loc_path, dr, f"{vid}_metadata.json"))
#                     off_x = drone_centres[(loc, dr)][0]
#                     off_y = drone_centres[(loc, dr)][1]
#                     ann_file = os.path.join(loc_path, dr, f"{vid}_annotations.txt")
#                     if not os.path.isfile(ann_file):
#                         continue
#                     df = pd.read_csv(ann_file, sep = ' ', header = None, 
#                                      names=['trackId','xmin','ymin','xmax','ymax','frame','lost','occluded','generated','label'])
#                     df['x'] = (df.xmin + df.xmax) / 2 + off_x
#                     df['y'] = (df.xmin + df.xmax) / 2 + off_y
#                     dfs.append(df[['trackId','frame','x','y','label', 'occluded']])
#                 if not dfs:
#                     continue
#                 data = pd.concat(dfs, ignore_index=True)
#                 data = data.drop_duplicates(subset=['trackId', 'frame'])
#                 for trackId, grp in data.groupby('trackId'):
#                     grp = grp.sort_values('frame')
#                     frames = grp['frame'].values
#                     coords = np.vstack([grp['x'].values, grp['y'].values]).T
#                     labels = grp['label'].values

#                     # sliding window
#                     L = len(frames)
#                     window = self.T_past + self.T_future
#                     for i in range(L - window + 1):
#                         # check continuous frames
#                         if frames[i+self.T_past-1] - frames[i] != self.T_past-1:
#                             continue
#                         if frames[i+window-1] - frames[i+self.T_past] != self.T_future-1:
#                             continue
#                         past = coords[i:i+self.T_past]
#                         future = coords[i+self.T_past:i+window]
#                         # label at last past step
#                         lbl = labels[i+self.T_past-1]
#                         if lbl not in self.cls2idx:
#                             continue
#                         self.samples.append({
#                             'past': past.astype(np.float32),
#                             'future': future.astype(np.float32),
#                             'label': self.cls2idx[lbl]
#                         })
                
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         s = self.samples[idx]
#         return {
#             'past':   torch.from_numpy(s['past']),    # (T_past, 2)
#             'future': torch.from_numpy(s['future']),  # (T_future, 2)
#             'label':  torch.tensor(s['label'], dtype=torch.int)
#         }
    

# class SequenceTransformer(nn.Module):
#     """
#     Simple Transformer that encodes past (x,y) sequences and
#     predicts future means and log-variances per time step.
#     """
#     def __init__(
#         self,
#         T_past: int,
#         T_future: int,
#         d_model: int = 128,
#         nhead: int = 4,
#         num_layers: int = 3,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         self.T_past = T_past
#         self.T_future = T_future
#         # project 2D coords into d_model
#         self.input_proj = nn.Linear(2, d_model)
#         # learned positional embeddings for encoder
#         self.pos_embed = nn.Parameter(torch.randn(T_past, d_model))
#         # transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=d_model*4,
#             dropout=dropout,
#             activation='gelu'
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # MLP head: from flattened encoder output to 4*T_future
#         self.head = nn.Sequential(
#             nn.LayerNorm(d_model * T_past),
#             nn.Linear(d_model * T_past, T_future * 4)
#         )

#     def forward(self, past: torch.Tensor):
#         """
#         Args:
#             past: (B, T_past, 2) tensor of past coords
#         Returns:
#             mu:     (B, T_future, 2)
#             logvar: (B, T_future, 2)
#         """
#         B = past.size(0)
#         # project
#         x = self.input_proj(past)              # (B, T_past, d_model)
#         # add positional embeddings
#         x = x + self.pos_embed.unsqueeze(0)    # broadcast to (B, T_past, d_model)
#         # transformer expects (seq, batch, feature)
#         x = x.permute(1, 0, 2)                 # (T_past, B, d_model)
#         x = self.transformer(x)               # (T_past, B, d_model)
#         x = x.permute(1, 0, 2).contiguous()    # (B, T_past, d_model)
#         # flatten time+feature
#         x = x.view(B, -1)                     # (B, T_past * d_model)
#         params = self.head(x)                 # (B, T_future * 4)
#         params = params.view(B, self.T_future, 4)
#         mu = params[..., :2]
#         logvar = params[..., 2:]
#         return mu, logvar

# def gaussian_nll_loss(mu, logvar, target):
#     """
#     Negative log-likelihood for diagonal Gaussian.
#     mu, logvar: (B, T_future, 2)
#     target:     (B, T_future, 2)
#     """
#     var = torch.exp(logvar)
#     # 0.5*(logvar + (target-mu)^2/var)
#     nll = 0.5 * (logvar + (target - mu)**2 / var)
#     # sum over x,y and time
#     return nll.sum(dim=[1, 2]).mean()

# # Example usage:
# if __name__ == '__main__':
#     # params
#     drone_root = 'square_stanford_data'
#     orig_root = 'stanford_campus_dataset'
#     classes = ['Pedestrian','Biker','Skater','Cart','Car','Bus']
#     T_past, T_future = 10, 10
#     # dataset & loader
#     ds = AgentSequenceDataset(drone_root, orig_root, classes, T_past, T_future)
#     loader = DataLoader(ds, batch_size=32, shuffle=True)
#     # model, optimizer
#     model = SequenceTransformer(T_past, T_future)
#     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#     # training loop
#     model.train()
#     for epoch in range(10):
#         total_loss = 0.0
#         for batch in loader:
#             past = batch['past']      # (B, T_past, 2)
#             future = batch['future']  # (B, T_future, 2)
#             mu, logvar = model(past)
#             loss = gaussian_nll_loss(mu, logvar, future)
#             opt.zero_grad(); loss.backward(); opt.step()
#             total_loss += loss.item() * past.size(0)
#         avg = total_loss / len(ds)
#         print(f"Epoch {epoch+1}, Loss = {avg:.4f}")

# agent_sequence_dataset.py



# import os
# import numpy as np
# import pandas as pd
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset

# class AgentSequenceDataset(Dataset):
#     """
#     PyTorch Dataset yielding per-agent trajectory sequences, with occlusion masks.

#     For each agent track, returns:
#       - past:   Tensor[T_past, 2]     (# x,y positions)
#       - future: Tensor[T_future, 2]   (# x,y positions)
#       - label:  Tensor                (# class index)
#       - mask:   Tensor[T_future]      (# 1=visible,0=occluded)
#     Only sequences where both past and future windows are continuous in time are included.
#     """
#     def __init__(self, drone_data_root, original_dataset_root, classes, T_past=10, T_future=10):
#         self.drone_data_root = drone_data_root
#         self.orig_root = original_dataset_root
#         self.classes = classes
#         self.cls2idx = {c: i for i, c in enumerate(classes)}
#         self.T_past = T_past
#         self.T_future = T_future
#         self.samples = []

#         # Preload drone centre offsets
#         drone_centres = {}
#         for loc in os.listdir(self.drone_data_root):
#             loc_path = os.path.join(self.drone_data_root, loc)
#             if not os.path.isdir(loc_path):
#                 continue
#             for dr in os.listdir(loc_path):
#                 meta_path = os.path.join(loc_path, dr, "video0_metadata.json")
#                 if not os.path.isfile(meta_path):
#                     continue
#                 with open(meta_path, 'r') as f:
#                     js = json.load(f)
#                 drone_centres[(loc, dr)] = (js["x_position"], js["y_position"])

#         # Build samples
#         for loc in os.listdir(self.drone_data_root):
#             loc_path = os.path.join(self.drone_data_root, loc)
#             if not os.path.isdir(loc_path):
#                 continue
#             drone1_dir = os.path.join(loc_path, "drone1")
#             vids = [fn.replace("_annotations.txt", "")
#                     for fn in os.listdir(drone1_dir) if fn.endswith("_annotations.txt")]
#             for vid in vids:
#                 dfs = []
#                 for dr in os.listdir(loc_path):
#                     if (loc, dr) not in drone_centres:
#                         continue
#                     off_x, off_y = drone_centres[(loc, dr)]
#                     ann_file = os.path.join(loc_path, dr, f"{vid}_annotations.txt")
#                     if not os.path.isfile(ann_file):
#                         continue
#                     df = pd.read_csv(
#                         ann_file, sep=' ', header=None,
#                         names=['trackId','xmin','ymin','xmax','ymax','frame',
#                                'lost','occluded','generated','label']
#                     )
#                     # convert local->global centre
#                     df['x'] = (df.xmin + df.xmax)/2 + off_x
#                     df['y'] = (df.ymin + df.ymax)/2 + off_y
#                     dfs.append(df[['trackId','frame','x','y','label','occluded']])
#                 if not dfs:
#                     continue
#                 data = pd.concat(dfs, ignore_index=True)
#                 data = data.drop_duplicates(subset=['trackId','frame'])

#                 # group by track
#                 for trackId, grp in data.groupby('trackId'):
#                     grp = grp.sort_values('frame')
#                     frames = grp['frame'].values
#                     coords = np.stack([grp['x'].values, grp['y'].values], axis=1)
#                     occls  = grp['occluded'].values.astype(int)
#                     labels = grp['label'].values

#                     L = len(frames)
#                     window = self.T_past + self.T_future
#                     for i in range(L - window + 1):
#                         # continuous in time?
#                         if frames[i + self.T_past - 1] - frames[i] != self.T_past - 1:
#                             continue
#                         if frames[i + window    - 1] - frames[i + self.T_past] != self.T_future - 1:
#                             continue
#                         past   = coords[i : i + self.T_past]
#                         future = coords[i + self.T_past : i + window]
#                         # occlusion mask for future
#                         mask = (occls[i + self.T_past : i + window] == 0).astype(np.float32)
#                         lbl = labels[i + self.T_past - 1]
#                         if lbl not in self.cls2idx:
#                             continue
#                         self.samples.append({
#                             'past':   past.astype(np.float32),
#                             'future': future.astype(np.float32),
#                             'label':  self.cls2idx[lbl],
#                             'mask':   mask
#                         })

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         s = self.samples[idx]
#         return {
#             'past':   torch.from_numpy(s['past']),    # (T_past, 2)
#             'future': torch.from_numpy(s['future']),  # (T_future, 2)
#             'label':  torch.tensor(s['label'], dtype=torch.long),
#             'mask':   torch.from_numpy(s['mask'])     # (T_future,)
#         }

# def gaussian_nll_loss(mu, logvar, target, mask):
#     """
#     Diagonal Gaussian NLL, ignoring occluded frames via mask.
#     mu, logvar: (B, T_future, 2)
#     target:     (B, T_future, 2)
#     mask:       (B, T_future) -> 1=visible, 0=occluded
#     """
#     var = torch.exp(logvar)
#     mask_exp = mask.unsqueeze(-1)   # (B, T_future, 1)
#     nll = 0.5 * (logvar + (target - mu)**2 / var)
#     nll = nll * mask_exp
#     total_vis = mask_exp.sum(dim=[1,2]).clamp_min(1.0)
#     loss = nll.sum(dim=[1,2]) / total_vis
#     return loss.mean()



# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len = 5000):
#         super().__init__()
#         pe = torch.zeroes(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         return x + self.pe[:x.size(0), :]
    
# class TrajectoryTransformer(nn.Module):
#     """
#     Autoregressive Transformer for trajectory prediction.
    
#     Takes past trajectories and autoregressively generates future positions
#     with Gaussian outputs (mean and log-variance).
#     """

#     def __init__(self, d_model:int, nhead: int, num_layers:int, dropout:int, max_seq_len=100):
#         super().__init__()
#         self.d_model = d_model
#         self.max_seq_len = max_seq_len
#         self.nhead = nhead
#         self.input_projection = nn.Linear(2, d_model)
#         self.dropout = dropout
#         self.num_layers = num_layers
#         self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.nhead,
#                                                    dim_feedforward=self.d_model * 4, dropout = self.dropout,
#                                                    activation = 'gelu', batch_first=False)
#         self.transformer = nn.TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = self.num_layers, norm = nn.LayerNorm(d_model))

#         self.mean_head = nn.Linear(d_model, 2)
#         self.variance = nn.Linear(d_model, 2)

#         self._init_weights()
#     def _init_weights(self):
#         for parameter in self.parameters:
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(parameter)

#     def create_causal_mask(self, seq_len):
#         mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
#         mask = mask.masked_fill(mask == 1, float('-inf'))
#         return mask
    
#     def forward(self, past_traj, future_traj=None, teacher_forcing=True):
#         """
#         Forward pass for training or inference.
        
#         Args:
#             past_traj: Tensor [batch_size, T_past, 2] - past trajectories
#             future_traj: Tensor [batch_size, T_future, 2] - future trajectories (for training)
#             teacher_forcing: bool - whether to use teacher forcing during training
            
#         Returns:
#             mu: Tensor [batch_size, T_future, 2] - predicted means
#             logvar: Tensor [batch_size, T_future, 2] - predicted log-variances
#         """
#         batch_size = past_traj.size(0)
#         T_past = past_traj.size(1)
        
#         if teacher_forcing and future_traj is not None:
#             # Training mode: use teacher forcing
#             T_future = future_traj.size(1)
            
#             # Concatenate past and future for teacher forcing
#             # Use past trajectory + future trajectory (shifted by 1 for autoregression)
#             full_traj = torch.cat([past_traj, future_traj], dim=1)  # [B, T_past+T_future, 2]
            
#             # Create input sequence for transformer
#             input_seq = full_traj[:, :-1, :]  # [B, T_past+T_future-1, 2]
#             seq_len = input_seq.size(1)
            
#             # Project to model dimension
#             input_seq = input_seq.transpose(0, 1)  # [seq_len, batch, 2]
#             x = self.input_projection(input_seq)  # [seq_len, batch, d_model]
            
#             # Add positional encoding
#             x = self.pos_encoding(x)
            
#             # Create causal mask
#             causal_mask = self.create_causal_mask(seq_len).to(x.device)
            
#             # Apply transformer
#             x = self.transformer(x, mask=causal_mask)
            
#             # Get predictions for future timesteps only
#             future_features = x[T_past:, :, :]  # [T_future, batch, d_model]
            
#             # Generate Gaussian parameters
#             mu = self.mean_head(future_features)  # [T_future, batch, 2]
#             logvar = self.logvar_head(future_features)  # [T_future, batch, 2]
            
#             # Transpose back to batch-first
#             mu = mu.transpose(0, 1)  # [batch, T_future, 2]
#             logvar = logvar.transpose(0, 1)  # [batch, T_future, 2]
            
#             return mu, logvar
        
#         else:
#             # Inference mode: autoregressive generation
#             T_future = future_traj.size(1) if future_traj is not None else 10
            
#             # Start with past trajectory
#             current_seq = past_traj.clone()  # [batch, T_past, 2]
            
#             mu_list = []
#             logvar_list = []
            
#             for t in range(T_future):
#                 # Prepare input sequence
#                 input_seq = current_seq.transpose(0, 1)  # [seq_len, batch, 2]
#                 seq_len = input_seq.size(0)
                
#                 # Project to model dimension
#                 x = self.input_projection(input_seq)  # [seq_len, batch, d_model]
                
#                 # Add positional encoding
#                 x = self.pos_encoding(x)
                
#                 # Create causal mask
#                 causal_mask = self.create_causal_mask(seq_len).to(x.device)
                
#                 # Apply transformer
#                 x = self.transformer(x, mask=causal_mask)
                
#                 # Get prediction for next timestep
#                 next_features = x[-1:, :, :]  # [1, batch, d_model]
                
#                 # Generate Gaussian parameters
#                 mu_t = self.mean_head(next_features)  # [1, batch, 2]
#                 logvar_t = self.logvar_head(next_features)  # [1, batch, 2]
                
#                 mu_list.append(mu_t.transpose(0, 1))  # [batch, 1, 2]
#                 logvar_list.append(logvar_t.transpose(0, 1))  # [batch, 1, 2]
                
#                 # Sample next position (or use mean for deterministic prediction)
#                 # For training stability, we use the mean
#                 next_pos = mu_t.transpose(0, 1)  # [batch, 1, 2]
                
#                 # Append to sequence
#                 current_seq = torch.cat([current_seq, next_pos], dim=1)
            
#             # Concatenate all predictions
#             mu = torch.cat(mu_list, dim=1)  # [batch, T_future, 2]
#             logvar = torch.cat(logvar_list, dim=1)  # [batch, T_future, 2]
            
#             return mu, logvar



# class TrajectoryPredictor(nn.Module):
#     """
#     Wrapper class that combines the transformer with the dataset's Gaussian NLL loss.
#     """
    
#     def __init__(self, 
#                  d_model=256,
#                  nhead=8,
#                  num_layers=6,
#                  dropout=0.1,
#                  max_seq_len=100):
#         super().__init__()
        
#         self.transformer = TrajectoryTransformer(
#             d_model=d_model,
#             nhead=nhead,
#             num_layers=num_layers,
#             dropout=dropout,
#             max_seq_len=max_seq_len
#         )
    
#     def forward(self, batch, teacher_forcing=True):
#         """
#         Forward pass using batch from AgentSequenceDataset.
        
#         Args:
#             batch: dict with keys 'past', 'future', 'label', 'mask'
#             teacher_forcing: bool - whether to use teacher forcing
            
#         Returns:
#             mu: predicted means
#             logvar: predicted log-variances
#             loss: Gaussian NLL loss
#         """
#         past_traj = batch['past']    # [batch_size, T_past, 2]
#         future_traj = batch['future']  # [batch_size, T_future, 2]
#         mask = batch['mask']         # [batch_size, T_future]
        
#         # Get predictions
#         mu, logvar = self.transformer(past_traj, future_traj, teacher_forcing)
        
#         # Compute Gaussian NLL loss
#         loss = gaussian_nll_loss(mu, logvar, future_traj, mask)
        
#         return mu, logvar, loss
    
#     def predict(self, past_traj, T_future=10):
#         """
#         Predict future trajectories given past trajectories.
        
#         Args:
#             past_traj: Tensor [batch_size, T_past, 2]
#             T_future: int - number of future timesteps to predict
            
#         Returns:
#             mu: predicted means [batch_size, T_future, 2]
#             logvar: predicted log-variances [batch_size, T_future, 2]
#         """
#         self.eval()
#         with torch.inference_mode():
#             # Create dummy future trajectory for shape inference
#             batch_size = past_traj.size(0)
#             dummy_future = torch.zeros(batch_size, T_future, 2, device=past_traj.device)
            
#             mu, logvar = self.transformer(past_traj, dummy_future, teacher_forcing=False)
            
#         return mu, logvar
    
# def create_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
#     model = TrajectoryPredictor(d_model = 128, nhead = 8, num_layers = 6, dropout = 0.1, max_seq_len = 100).to(device)
#     return model

# def train_model(model, train_loader, val_loader, epochs = 50, lr = 1e-4, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
#     optimizer = torch.optim.AdamW(model.parameters, lr=lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    



# agent_sequence_dataset_and_model.py

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class AgentSequenceDataset(Dataset):
    """
    PyTorch Dataset yielding per-agent trajectory sequences with:
      - past coords in global frame
      - observation mask (which past frames were seen by any drone)
      - future coords (ground truth)
      - occlusion mask for future (which future frames are not occluded)
      - label (class index)
    """
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
            if loc == 'hyang': continue
            for vid in os.listdir(os.path.join(original_dataset_root,"annotations",loc)):
                # load original annotations
                orig_file = os.path.join(original_dataset_root,"annotations",loc,vid,"annotations.txt")
                if not os.path.isfile(orig_file): continue
                odf = pd.read_csv(orig_file, sep='\s+', header=None,
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
            'label':    torch.tensor(s['label'],dtype=torch.long)
        }


class TrajectoryModel(nn.Module):
    """
    Transformer encoder + GRU-based autoregressive decoder predicting
    future Gaussian (mu, logvar) per timestep, with mask embedding.
    """
    def __init__(self, T_past, T_future, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.T_past = T_past
        self.T_future = T_future
        self.d_model = d_model

        # input projection for coords
        self.input_proj = nn.Linear(2, d_model)
        # mask embeddings
        self.e_obs  = nn.Parameter(torch.randn(d_model))
        self.e_miss = nn.Parameter(torch.randn(d_model))
        # positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(T_past, d_model))

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # decoder GRU cell
        self.gru = nn.GRUCell(d_model, d_model)
        # output head: d_model -> 4 dims
        self.out_head = nn.Linear(d_model, 4)

    def forward(self, past, obs_mask, future=None):
        # past: (B, T_past, 2), obs_mask: (B, T_past)
        B = past.size(0)
        # embed coords
        x = self.input_proj(past)  # (B,T_past,d)
        # add mask embedding
        mask_embed = obs_mask.unsqueeze(-1)*(self.e_obs) + (1-obs_mask).unsqueeze(-1)*(self.e_miss)
        x = x + mask_embed
        # add pos emb
        x = x + self.pos_emb.unsqueeze(0)
        # encoder
        x_t = x.permute(1,0,2)     # (T_past,B,d)
        h_enc = self.encoder(x_t)  # (T_past,B,d)
        h = h_enc.mean(0)          # (B,d)

        # autoregressive decode
        mu_seq, logvar_seq = [], []
        inp = None
        for t in range(self.T_future):
            # decide input to GRU
            if future is not None:
                coord = future[:,t]             # (B,2)
                inp = self.input_proj(coord)    # teacher forcing
            else:
                inp = mu_seq[-1]                # previous mu
            # update hidden
            h = self.gru(inp, h)               # (B,d)
            # predict params
            params = self.out_head(h)          # (B,4)
            mu_t = params[:,:2]
            logvar_t = params[:,2:]
            mu_seq.append(mu_t)
            logvar_seq.append(logvar_t)
        mu      = torch.stack(mu_seq, dim=1)      # (B, T_future,2)
        logvar  = torch.stack(logvar_seq, dim=1)  # (B, T_future,2)
        return mu, logvar


def gaussian_nll_loss(mu, logvar, target, occ_mask):
    """
    Diagonal Gaussian NLL, masked by occ_mask (1=visible).
    mu, logvar: (B, T_future,2)
    target:     (B, T_future,2)
    occ_mask:   (B, T_future)
    """
    var = torch.exp(logvar)
    diff2 = (target - mu)**2
    nll = 0.5*(logvar + diff2/var)
    mask = occ_mask.unsqueeze(-1)
    nll = nll * mask
    total_vis = mask.sum(dim=[1,2]).clamp_min(1.0)
    loss = nll.sum(dim=[1,2]) / total_vis
    return loss.mean()

# End of code file




