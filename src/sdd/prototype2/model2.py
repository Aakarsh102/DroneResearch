import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
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

class GraphInteractionModel(nn.Module):
    def __init__(self, num_classes, locations, d_model=256, nhead=8, num_layers=6, 
                 T_past=10, T_future=10, use_deltas=True, predict_uncertainty=True):
        super(GraphInteractionModel, self).__init__()
        
        self.d_model = d_model
        self.T_past = T_past
        self.T_future = T_future
        self.use_deltas = use_deltas
        self.num_classes = num_classes
        self.predict_uncertainty = predict_uncertainty
        self.nhead = nhead
        self.num_layers = num_layers

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

        # Add a learned token for "unobserved" positions
        self.unobserved_token = nn.Parameter(torch.randn(d_model))
        
        # Add temporal gap encoding - encodes how many frames since last observation
        self.gap_encoding = nn.Embedding(T_past + 1, d_model)  # +1 for gaps > T_past

        self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.nhead, dim_feedforward=4 * self.d_model,
                                                        activation='gelu', batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.d_model * 4,
                                                        activation='gelu', batch_first=True)
        self.decoder_transformer = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)

        self.position_interpolator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        if use_deltas:
            if predict_uncertainty:
                self.delta_head = nn.Linear(d_model, 4)  # mu_x, mu_y, logvar_x, logvar_y
            else:
                self.delta_head = nn.Linear(d_model, 2)
        
        if predict_uncertainty:
            self.position_head = nn.Linear(d_model, 4)  # mu_x, mu_y, logvar_x, logvar_y
        else:
            self.position_head = nn.Linear(d_model, 2)

    


