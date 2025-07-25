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

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len = 1000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]


# class SpatialEncoding(nn.Module):
#     def __init__(self, d_model, max_pos=10000):
#         super(SpatialEncoding, self).__init__()
#         self.d_model = d_model
#         self.max_pos = max_pos
#         self.pos_embed = nn.Linear(2, d_model)
#         self.half_dim = d_model//2

#         div_term = torch.exp(
#             torch.arange(0, self.half_dim, 2).float() *
#             ( -math.log(10000.0) / self.half_dim )
#         )
#         self.register_buffer('div_term', div_term)

#     def forward(self, positions: torch.Tensor) -> torch.Tensor:
#         """
#         positions: (B, T, 2)  — normalized x,y in [0,1]
#         returns:    (B, T, d_model)
#         """
#         B, T, _ = positions.shape
#         scaled = positions * self.max_pos   # now in [0, max_pos]

#         pe_x = torch.zeros(B, T, self.half_dim, device=positions.device)

#         pe_x[:, :, 0::2] = torch.sin(scaled[:, :, 0:1] * self.div_term)
#         pe_x[:, :, 1::2] = torch.cos(scaled[:, :, 0:1] * self.div_term)

#         pe_y = torch.zeros(B, T, self.half_dim, device=positions.device)
#         pe_y[:, :, 0::2] = torch.sin(scaled[:, :, 1:2] * self.div_term)
#         pe_y[:, :, 1::2] = torch.cos(scaled[:, :, 1:2] * self.div_term)
        
#         # concatenate x and y → (B, T, d_model)
#         return torch.cat([pe_x, pe_y], dim=-1)

# class GraphInteractionModel(nn.Module):
#     def __init__(self, num_classes, locations, d_model=256, nhead=8, num_layers=6, 
#                  T_past=10, T_future=10, use_deltas=True, predict_uncertainty=True):
#         super(GraphInteractionModel, self).__init__()
        
#         self.d_model = d_model
#         self.T_past = T_past
#         self.T_future = T_future
#         self.use_deltas = use_deltas
#         self.num_classes = num_classes
#         self.predict_uncertainty = predict_uncertainty
#         self.nhead = nhead
#         self.num_layers = num_layers

#         self.location_embedding = nn.Embedding(len(locations), d_model)
#         self.location_to_idx = {loc:i for i,loc in enumerate(locations)}

#         self.class_embedding = nn.Embedding(num_classes, d_model)
        
#         # Spatial encoding for absolute positions (uses original coordinates)
#         self.spatial_encoding = SpatialEncoding(d_model)
        
#         # Temporal encoding
#         self.temporal_encoding = PositionalEncoding(d_model)

#         self.pos_projection = nn.Linear(2, d_model)
#         if use_deltas:
#             self.delta_projection = nn.Linear(2, d_model)

#         # Add a learned token for "unobserved" positions
#         self.unobserved_token = nn.Parameter(torch.randn(d_model))
        
#         # Add temporal gap encoding - encodes how many frames since last observation
#         self.gap_encoding = nn.Embedding(T_past + 1, d_model)  # +1 for gaps > T_past

#         self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.nhead, dim_feedforward=4 * self.d_model,
#                                                         activation='gelu', batch_first=True)
#         self.encoder_transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

#         self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.d_model * 4,
#                                                         activation='gelu', batch_first=True)
#         self.decoder_transformer = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)

#         self.position_interpolator = nn.Sequential(
#             nn.Linear(d_model, d_model // 2),
#             nn.ReLU(),
#             nn.Linear(d_model // 2, 2)
#         )

#         if use_deltas:
#             if predict_uncertainty:
#                 self.delta_head = nn.Linear(d_model, 4)  # mu_x, mu_y, logvar_x, logvar_y
#             else:
#                 self.delta_head = nn.Linear(d_model, 2)
        
#         if predict_uncertainty:
#             self.position_head = nn.Linear(d_model, 4)  # mu_x, mu_y, logvar_x, logvar_y
#         else:
#             self.position_head = nn.Linear(d_model, 2)

    

import torch
import torch.nn as nn
import math

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
        positions: (..., 2)  — normalized x,y in [0,1]
        returns:    (..., d_model)
        """
        original_shape = positions.shape
        positions_flat = positions.view(-1, 2)
        scaled = positions_flat * self.max_pos   # now in [0, max_pos]

        pe_x = torch.zeros(positions_flat.size(0), self.half_dim, device=positions.device)
        pe_x[:, 0::2] = torch.sin(scaled[:, 0:1] * self.div_term)
        pe_x[:, 1::2] = torch.cos(scaled[:, 0:1] * self.div_term)

        pe_y = torch.zeros(positions_flat.size(0), self.half_dim, device=positions.device)
        pe_y[:, 0::2] = torch.sin(scaled[:, 1:2] * self.div_term)
        pe_y[:, 1::2] = torch.cos(scaled[:, 1:2] * self.div_term)
        
        pe = torch.cat([pe_x, pe_y], dim=-1)  # (flattened, d_model)
        return pe.view(*original_shape[:-1], self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)

class AlternatingTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(AlternatingTransformerLayer, self).__init__()
        self.temporal_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.agent_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        
    def forward(self, x, layer_idx, attention_mask=None):
        """
        x: input tensor
        layer_idx: which layer this is (0-indexed)
        attention_mask: mask for attention
        """
        if layer_idx % 2 == 0:
            # Even layers: temporal attention (B*N, T, d_model)
            return self.temporal_layer(x, src_key_padding_mask=attention_mask)
        else:
            # Odd layers: agent attention (B*T, N, d_model)
            return self.agent_layer(x, src_key_padding_mask=attention_mask)

class AlternatingTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(AlternatingTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            AlternatingTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
    def forward(self, x, batch_size, num_agents, seq_len, attention_mask=None):
        """
        x: (B*N, T, d_model) initial format
        attention_mask: (B*N, T) or (B*T, N) depending on layer
        """
        current_x = x
        current_mask = attention_mask
        
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                # Even layer: temporal attention (B*N, T, d_model)
                if current_x.shape != (batch_size * num_agents, seq_len, x.size(-1)):
                    # Reshape from (B*T, N, d_model) to (B*N, T, d_model)
                    current_x = current_x.view(batch_size, seq_len, num_agents, -1)
                    current_x = current_x.permute(0, 2, 1, 3).contiguous()
                    current_x = current_x.view(batch_size * num_agents, seq_len, -1)
                    
                    if current_mask is not None:
                        current_mask = current_mask.view(batch_size, seq_len, num_agents)
                        current_mask = current_mask.permute(0, 2, 1).contiguous()
                        current_mask = current_mask.view(batch_size * num_agents, seq_len)
                        
            else:
                # Odd layer: agent attention (B*T, N, d_model)
                if current_x.shape != (batch_size * seq_len, num_agents, x.size(-1)):
                    # Reshape from (B*N, T, d_model) to (B*T, N, d_model)
                    current_x = current_x.view(batch_size, num_agents, seq_len, -1)
                    current_x = current_x.permute(0, 2, 1, 3).contiguous()
                    current_x = current_x.view(batch_size * seq_len, num_agents, -1)
                    
                    if current_mask is not None:
                        current_mask = current_mask.view(batch_size, num_agents, seq_len)
                        current_mask = current_mask.permute(0, 2, 1).contiguous()
                        current_mask = current_mask.view(batch_size * seq_len, num_agents)
            
            current_x = layer(current_x, i, current_mask)
        
        # Return in (B*N, T, d_model) format for consistency
        if current_x.shape[0] == batch_size * seq_len:  # Currently (B*T, N, d_model)
            current_x = current_x.view(batch_size, seq_len, num_agents, -1)
            current_x = current_x.permute(0, 2, 1, 3).contiguous()
            current_x = current_x.view(batch_size * num_agents, seq_len, -1)
            
        return current_x

class AlternatingTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(AlternatingTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            self._create_decoder_layer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
    def _create_decoder_layer(self, d_model, nhead, dim_feedforward, dropout):
        return nn.ModuleDict({
            'temporal': nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            ),
            'agent': nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
        })
        
    def forward(self, tgt, memory, batch_size, num_agents, tgt_seq_len, mem_seq_len,
                tgt_mask=None, memory_key_padding_mask=None):
        """
        tgt: (B*N, T_future, d_model) initial format
        memory: (B*N, T_past, d_model)
        """
        current_tgt = tgt
        current_memory = memory
        current_tgt_mask = tgt_mask
        current_mem_mask = memory_key_padding_mask
        
        for i, layer_dict in enumerate(self.layers):
            if i % 2 == 0:
                # Even layer: temporal attention
                layer = layer_dict['temporal']
                
                # Ensure correct format (B*N, T, d_model)
                if current_tgt.shape[0] == batch_size * tgt_seq_len:
                    current_tgt = current_tgt.view(batch_size, tgt_seq_len, num_agents, -1)
                    current_tgt = current_tgt.permute(0, 2, 1, 3).contiguous()
                    current_tgt = current_tgt.view(batch_size * num_agents, tgt_seq_len, -1)
                    
                if current_memory.shape[0] == batch_size * mem_seq_len:
                    current_memory = current_memory.view(batch_size, mem_seq_len, num_agents, -1)
                    current_memory = current_memory.permute(0, 2, 1, 3).contiguous()
                    current_memory = current_memory.view(batch_size * num_agents, mem_seq_len, -1)
                    
                # Handle masks
                if current_mem_mask is not None and current_mem_mask.shape[0] == batch_size * mem_seq_len:
                    current_mem_mask = current_mem_mask.view(batch_size, mem_seq_len, num_agents)
                    current_mem_mask = current_mem_mask.permute(0, 2, 1).contiguous()
                    current_mem_mask = current_mem_mask.view(batch_size * num_agents, mem_seq_len)
                    
            else:
                # Odd layer: agent attention
                layer = layer_dict['agent']
                
                # Ensure correct format (B*T, N, d_model)
                if current_tgt.shape[0] == batch_size * num_agents:
                    current_tgt = current_tgt.view(batch_size, num_agents, tgt_seq_len, -1)
                    current_tgt = current_tgt.permute(0, 2, 1, 3).contiguous()
                    current_tgt = current_tgt.view(batch_size * tgt_seq_len, num_agents, -1)
                    
                if current_memory.shape[0] == batch_size * num_agents:
                    current_memory = current_memory.view(batch_size, num_agents, mem_seq_len, -1)
                    current_memory = current_memory.permute(0, 2, 1, 3).contiguous()
                    current_memory = current_memory.view(batch_size * mem_seq_len, num_agents, -1)
                    
                # Handle masks  
                if current_mem_mask is not None and current_mem_mask.shape[0] == batch_size * num_agents:
                    current_mem_mask = current_mem_mask.view(batch_size, num_agents, mem_seq_len)
                    current_mem_mask = current_mem_mask.permute(0, 2, 1).contiguous()
                    current_mem_mask = current_mem_mask.view(batch_size * mem_seq_len, num_agents)
            
            current_tgt = layer(
                tgt=current_tgt,
                memory=current_memory,
                tgt_mask=current_tgt_mask,
                memory_key_padding_mask=current_mem_mask
            )
        
        # Return in (B*N, T_future, d_model) format for consistency
        if current_tgt.shape[0] == batch_size * tgt_seq_len:
            current_tgt = current_tgt.view(batch_size, tgt_seq_len, num_agents, -1)
            current_tgt = current_tgt.permute(0, 2, 1, 3).contiguous()
            current_tgt = current_tgt.view(batch_size * num_agents, tgt_seq_len, -1)
            
        return current_tgt

class GraphInteractionModel(nn.Module):
    def __init__(self, num_classes, locations, d_model=256, nhead=8, num_layers=6,
                 T_past=10, T_future=10, max_agents=50):
        super(GraphInteractionModel, self).__init__()
        
        self.d_model = d_model
        self.T_past = T_past
        self.T_future = T_future
        self.num_classes = num_classes
        self.max_agents = max_agents

        # Embeddings
        self.location_embedding = nn.Embedding(len(locations), d_model)
        self.location_to_idx = {loc: i for i, loc in enumerate(locations)}
        self.class_embedding = nn.Embedding(num_classes, d_model)
        
        # Agent ID embedding for distinguishing agents
        self.agent_embedding = nn.Embedding(max_agents, d_model)
        
        # Spatial and temporal encodings
        self.spatial_encoding = SpatialEncoding(d_model)
        self.temporal_encoding = PositionalEncoding(d_model)
        
        # Position projections
        self.pos_projection = nn.Linear(2, d_model)
        
        # Special tokens
        self.unobserved_token = nn.Parameter(torch.randn(d_model))
        self.gap_encoding = nn.Embedding(T_past + 1, d_model)
        
        # Alternating transformer layers
        self.transformer_encoder = AlternatingTransformerEncoder(
            d_model, nhead, num_layers, d_model * 4
        )
        
        self.transformer_decoder = AlternatingTransformerDecoder(
            d_model, nhead, num_layers, d_model * 4
        )
        
        # Output heads - predict delta_x, delta_y, var_x, var_y (not log variance)
        self.delta_head = nn.Linear(d_model, 4)  # mu_x, mu_y, var_x, var_y
        self.var_activation = nn.Softplus()  # Ensures positive variance
        
        # Position interpolation for current position estimation
        self.position_interpolator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, batch, use_teacher_forcing=True):
        """
        batch contains:
        - past_positions: (B, N, T_past, 2)
        - past_positions_orig: (B, N, T_past, 2) 
        - obs_mask: (B, N, T_past)
        - future_positions: (B, N, T_future, 2) [if training]
        - location: list of location strings, length B
        - label: (B, N) class labels for each agent
        """
        batch_size, num_agents = batch['past_positions'].shape[:2]
        device = batch['past_positions'].device
        
        # Get embeddings
        location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], 
                                      device=device)
        location_emb = self.location_embedding(location_indices)  # (B, d_model)
        class_emb = self.class_embedding(batch['label'])  # (B, N, d_model)
        
        # Agent embeddings
        agent_ids = torch.arange(num_agents, device=device).unsqueeze(0).expand(batch_size, -1)
        agent_emb = self.agent_embedding(agent_ids)  # (B, N, d_model)
        
        # Create enhanced past embeddings
        past_emb = self._create_enhanced_past_embeddings(
            batch['past_positions'], batch['past_positions_orig'], batch['obs_mask'],
            location_emb, class_emb, agent_emb, device
        )  # (B*N, T_past, d_model)
        
        # Create attention mask
        obs_mask_flat = batch['obs_mask'].view(batch_size * num_agents, self.T_past)
        attention_mask = (obs_mask_flat == 0)  # True where padded
        
        # Encode with alternating transformer
        memory = self.transformer_encoder(
            past_emb, batch_size, num_agents, self.T_past, attention_mask
        )  # (B*N, T_past, d_model)
        
        # Estimate current positions
        current_pos_est = self._estimate_current_position(
            memory, batch['obs_mask'], batch['past_positions']
        )  # (B, N, 2)
        
        # Generate future predictions
        if use_teacher_forcing and self.training:
            future_predictions = self._teacher_forcing_predictions(
                memory, attention_mask, current_pos_est, batch['future_positions'],
                location_emb, class_emb, agent_emb, batch_size, num_agents, device
            )
        else:
            future_predictions = self._generate_autoregressive_predictions(
                memory, attention_mask, current_pos_est, location_emb, class_emb, 
                agent_emb, batch_size, num_agents, device
            )
        
        outputs = self._process_outputs(future_predictions)
        outputs['current_position_estimate'] = current_pos_est
        
        return outputs
    
    def _create_enhanced_past_embeddings(self, past_positions, past_positions_orig, obs_mask,
                                       location_emb, class_emb, agent_emb, device):
        """Create enhanced embeddings for past trajectory"""
        batch_size, num_agents, seq_len, _ = past_positions.shape
        
        # Calculate gaps (time since last observation)
        gaps = torch.zeros_like(obs_mask, dtype=torch.long)
        for b in range(batch_size):
            for n in range(num_agents):
                gap_counter = 0
                for t in range(seq_len):
                    if obs_mask[b, n, t] == 1:
                        gap_counter = 0
                    else:
                        gap_counter += 1
                    gaps[b, n, t] = min(gap_counter, self.T_past)
        
        # Create position embeddings
        pos_emb = torch.zeros(batch_size, num_agents, seq_len, self.d_model, device=device)
        
        for b in range(batch_size):
            for n in range(num_agents):
                for t in range(seq_len):
                    if obs_mask[b, n, t] == 1:
                        pos_emb[b, n, t] = self.pos_projection(past_positions[b, n, t])
                    else:
                        pos_emb[b, n, t] = self.unobserved_token
        
        # Add spatial encoding (only for observed positions)
        spatial_emb = self.spatial_encoding(past_positions_orig)  # (B, N, T, d_model)
        obs_mask_expanded = obs_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
        pos_emb = pos_emb + spatial_emb * obs_mask_expanded
        
        # Add gap encoding
        gap_emb = self.gap_encoding(gaps)  # (B, N, T, d_model)
        pos_emb = pos_emb + gap_emb
        
        # Add context embeddings (location, class, agent)
        location_expanded = location_emb.unsqueeze(1).unsqueeze(2).expand(-1, num_agents, seq_len, -1)
        class_expanded = class_emb.unsqueeze(2).expand(-1, -1, seq_len, -1)
        agent_expanded = agent_emb.unsqueeze(2).expand(-1, -1, seq_len, -1)
        
        pos_emb = pos_emb + location_expanded + class_expanded + agent_expanded
        
        # Reshape to (B*N, T, d_model) for transformer processing
        pos_emb = pos_emb.view(batch_size * num_agents, seq_len, self.d_model)
        
        # Add temporal encoding
        pos_emb = self.temporal_encoding(pos_emb)
        
        return pos_emb
    
    def _estimate_current_position(self, memory, obs_mask, past_positions):
        """Estimate current position for each agent"""
        batch_size, num_agents, seq_len, _ = past_positions.shape
        
        # Reshape memory to (B, N, T, d_model)
        memory_reshaped = memory.view(batch_size, num_agents, seq_len, self.d_model)
        
        # Attention-weighted combination
        weights = torch.softmax(memory_reshaped.mean(dim=-1), dim=-1)  # (B, N, T)
        weights = weights * obs_mask  # Zero out unobserved positions
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
        
        # Weighted combination of memory states
        context = torch.sum(memory_reshaped * weights.unsqueeze(-1), dim=2)  # (B, N, d_model)
        
        # Predict current position
        current_pos = self.position_interpolator(context)  # (B, N, 2)
        
        return current_pos
    
    def _teacher_forcing_predictions(self, memory, attention_mask, current_pos_est,
                                   future_positions_target, location_emb, class_emb, agent_emb,
                                   batch_size, num_agents, device):
        """Teacher forcing for training"""
        # Flatten current position estimate: (B, N, 2) -> (B*N, 2)
        current_pos_flat = current_pos_est.view(batch_size * num_agents, 2)
        
        # Flatten future positions: (B, N, T_future, 2) -> (B*N, T_future, 2)
        future_pos_flat = future_positions_target.view(batch_size * num_agents, self.T_future, 2)
        
        # Create decoder input sequence: [current_pos, future_pos[0], ..., future_pos[T_future-2]]
        decoder_positions = torch.cat([
            current_pos_flat.unsqueeze(1),  # (B*N, 1, 2)
            future_pos_flat[:, :-1, :]     # (B*N, T_future-1, 2)
        ], dim=1)  # (B*N, T_future, 2)
        
        # Project to embedding space
        decoder_input = self.pos_projection(decoder_positions)  # (B*N, T_future, d_model)
        
        # Add context embeddings
        location_flat = location_emb.unsqueeze(1).expand(batch_size, num_agents, -1).contiguous()
        location_flat = location_flat.view(batch_size * num_agents, self.d_model)
        class_flat = class_emb.view(batch_size * num_agents, self.d_model)
        agent_flat = agent_emb.view(batch_size * num_agents, self.d_model)
        
        context_emb = location_flat + class_flat + agent_flat  # (B*N, d_model)
        decoder_input = decoder_input + context_emb.unsqueeze(1)
        
        # Add temporal encoding for future timesteps
        for i in range(self.T_future):
            temporal_pos = self.T_past + i
            if temporal_pos < self.temporal_encoding.pe.size(0):
                temporal_emb = self.temporal_encoding.pe[temporal_pos]  # (d_model,)
                decoder_input[:, i, :] += temporal_emb
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(self.T_future, self.T_future, device=device) * float('-inf'), 
            diagonal=1
        )
        
        # Apply alternating transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory,
            batch_size=batch_size,
            num_agents=num_agents,
            tgt_seq_len=self.T_future,
            mem_seq_len=self.T_past,
            tgt_mask=causal_mask,
            memory_key_padding_mask=attention_mask
        )  # (B*N, T_future, d_model)
        
        # Generate delta predictions
        delta_predictions = self.delta_head(decoder_output)  # (B*N, T_future, 4)
        
        # Split into mean and variance
        delta_mu = delta_predictions[..., :2]  # (B*N, T_future, 2)
        delta_var = self.var_activation(delta_predictions[..., 2:])  # (B*N, T_future, 2)
        
        # Convert deltas to positions
        positions_mu = self._convert_deltas_to_positions_teacher_forcing(
            current_pos_flat, delta_mu
        )  # (B*N, T_future, 2)
        
        # Propagate uncertainty
        positions_var = self._propagate_variance(delta_var)  # (B*N, T_future, 2)
        
        return {
            'future_deltas_mu': delta_mu.view(batch_size, num_agents, self.T_future, 2),
            'future_deltas_var': delta_var.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_mu': positions_mu.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_var': positions_var.view(batch_size, num_agents, self.T_future, 2)
        }
    
    def _convert_deltas_to_positions_teacher_forcing(self, start_pos, deltas):
        """Convert delta predictions to positions during teacher forcing"""
        batch_size_agents, seq_len, _ = deltas.shape
        positions = torch.zeros_like(deltas)
        
        current_pos = start_pos
        for i in range(seq_len):
            current_pos = current_pos + deltas[:, i, :]
            positions[:, i, :] = current_pos
        
        return positions
    
    def _propagate_variance(self, delta_var):
        """Propagate variance through cumulative sum"""
        batch_size_agents, seq_len, _ = delta_var.shape
        
        # Cumulative variance propagation
        pos_var = torch.zeros_like(delta_var)
        pos_var[:, 0] = delta_var[:, 0]
        
        for i in range(1, seq_len):
            # Variance accumulates when integrating deltas
            pos_var[:, i] = pos_var[:, i-1] + delta_var[:, i]
        
        return pos_var
    
    def _generate_autoregressive_predictions(self, memory, attention_mask, current_pos_est,
                                           location_emb, class_emb, agent_emb, 
                                           batch_size, num_agents, device):
        """Generate future predictions autoregressively"""
        # Flatten current position: (B, N, 2) -> (B*N, 2)
        current_pos = current_pos_est.view(batch_size * num_agents, 2)
        
        # Context embeddings
        location_flat = location_emb.unsqueeze(1).expand(batch_size, num_agents, -1).contiguous()
        location_flat = location_flat.view(batch_size * num_agents, self.d_model)
        class_flat = class_emb.view(batch_size * num_agents, self.d_model)
        agent_flat = agent_emb.view(batch_size * num_agents, self.d_model)
        context_emb = location_flat + class_flat + agent_flat
        
        # Storage for predictions
        future_deltas_mu = []
        future_deltas_var = []
        future_positions = []
        
        for t in range(self.T_future):
            # Prepare decoder input for current timestep
            decoder_input = self.pos_projection(current_pos)  # (B*N, d_model)
            decoder_input = decoder_input.unsqueeze(1)  # (B*N, 1, d_model)
            
            # Add context embeddings
            decoder_input = decoder_input + context_emb.unsqueeze(1)
            
            # Add temporal encoding
            temporal_pos = self.T_past + t
            if temporal_pos < self.temporal_encoding.pe.size(0):
                temporal_emb = self.temporal_encoding.pe[temporal_pos]
                decoder_input = decoder_input + temporal_emb.unsqueeze(0).unsqueeze(0)
            
            # Apply transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=memory,
                batch_size=batch_size,
                num_agents=num_agents,
                tgt_seq_len=1,
                mem_seq_len=self.T_past,
                memory_key_padding_mask=attention_mask
            )  # (B*N, 1, d_model)
            
            # Generate predictions
            delta_pred = self.delta_head(decoder_output.squeeze(1))  # (B*N, 4)
            delta_mu = delta_pred[..., :2]  # (B*N, 2)
            delta_var = self.var_activation(delta_pred[..., 2:])  # (B*N, 2)
            
            # Update position (use mean for next step)
            next_pos = current_pos + delta_mu
            
            # Store predictions
            future_deltas_mu.append(delta_mu)
            future_deltas_var.append(delta_var)
            future_positions.append(next_pos)
            
            # Update current position for next iteration
            current_pos = next_pos
        
        # Stack predictions
        deltas_mu = torch.stack(future_deltas_mu, dim=1)  # (B*N, T_future, 2)
        deltas_var = torch.stack(future_deltas_var, dim=1)  # (B*N, T_future, 2)
        positions = torch.stack(future_positions, dim=1)  # (B*N, T_future, 2)
        
        # Propagate variance for positions
        positions_var = self._propagate_variance(deltas_var)  # (B*N, T_future, 2)
        
        return {
            'future_deltas_mu': deltas_mu.view(batch_size, num_agents, self.T_future, 2),
            'future_deltas_var': deltas_var.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_mu': positions.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_var': positions_var.view(batch_size, num_agents, self.T_future, 2)
        }
    
    def _process_outputs(self, predictions):
        """Process predictions into final output format"""
        return predictions
    
    def predict_autoregressive(self, batch, sampling_strategy='mean', temperature=1.0):
        """
        Inference-time autoregressive prediction with sampling options
        
        Args:
            batch: Input batch
            sampling_strategy: 'mean', 'sample'
            temperature: Temperature for sampling
        """
        self.eval()
        with torch.no_grad():
            batch_size, num_agents = batch['past_positions'].shape[:2]
            device = batch['past_positions'].device
            
            # Get embeddings
            location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], 
                                          device=device)
            location_emb = self.location_embedding(location_indices)
            class_emb = self.class_embedding(batch['label'])
            
            agent_ids = torch.arange(num_agents, device=device).unsqueeze(0).expand(batch_size, -1)
            agent_emb = self.agent_embedding(agent_ids)
            
            # Create enhanced past embeddings
            past_emb = self._create_enhanced_past_embeddings(
                batch['past_positions'], batch['past_positions_orig'], batch['obs_mask'],
                location_emb, class_emb, agent_emb, device
            )
            
            # Create attention mask
            obs_mask_flat = batch['obs_mask'].view(batch_size * num_agents, self.T_past)
            attention_mask = (obs_mask_flat == 0)
            
            # Encode
            memory = self.transformer_encoder(
                past_emb, batch_size, num_agents, self.T_past, attention_mask
            )
            
            # Estimate current positions
            current_pos_est = self._estimate_current_position(
                memory, batch['obs_mask'], batch['past_positions']
            )
            
            # Generate predictions with sampling
            if sampling_strategy == 'mean':
                predictions = self._generate_autoregressive_predictions(
                    memory, attention_mask, current_pos_est, location_emb, class_emb,
                    agent_emb, batch_size, num_agents, device
                )
            else:
                predictions = self._generate_autoregressive_predictions_with_sampling(
                    memory, attention_mask, current_pos_est, location_emb, class_emb,
                    agent_emb, batch_size, num_agents, device, temperature
                )
            
            outputs = self._process_outputs(predictions)
            outputs['current_position_estimate'] = current_pos_est
            
            return outputs
    
    def _generate_autoregressive_predictions_with_sampling(self, memory, attention_mask, 
                                                         current_pos_est, location_emb, class_emb,
                                                         agent_emb, batch_size, num_agents, 
                                                         device, temperature):
        """Generate predictions with uncertainty sampling"""
        # Flatten current position
        current_pos = current_pos_est.view(batch_size * num_agents, 2)
        
        # Context embeddings
        location_flat = location_emb.unsqueeze(1).expand(batch_size, num_agents, -1).contiguous()
        location_flat = location_flat.view(batch_size * num_agents, self.d_model)
        class_flat = class_emb.view(batch_size * num_agents, self.d_model)
        agent_flat = agent_emb.view(batch_size * num_agents, self.d_model)
        context_emb = location_flat + class_flat + agent_flat
        
        # Storage for predictions
        future_deltas_mu = []
        future_deltas_var = []
        future_positions = []
        
        for t in range(self.T_future):
            # Prepare decoder input
            decoder_input = self.pos_projection(current_pos).unsqueeze(1)
            decoder_input = decoder_input + context_emb.unsqueeze(1)
            
            # Add temporal encoding
            temporal_pos = self.T_past + t
            if temporal_pos < self.temporal_encoding.pe.size(0):
                temporal_emb = self.temporal_encoding.pe[temporal_pos]
                decoder_input = decoder_input + temporal_emb.unsqueeze(0).unsqueeze(0)
            
            # Apply transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=memory,
                batch_size=batch_size,
                num_agents=num_agents,
                tgt_seq_len=1,
                mem_seq_len=self.T_past,
                memory_key_padding_mask=attention_mask
            )
            
            # Generate predictions
            delta_pred = self.delta_head(decoder_output.squeeze(1))
            delta_mu = delta_pred[..., :2]
            delta_var = self.var_activation(delta_pred[..., 2:])
            
            # Sample from the distribution
            delta_std = torch.sqrt(delta_var)
            eps = torch.randn_like(delta_mu)
            delta_sampled = delta_mu + temperature * delta_std * eps
            
            # Update position
            next_pos = current_pos + delta_sampled
            
            # Store predictions (store the parameters, not the samples)
            future_deltas_mu.append(delta_mu)
            future_deltas_var.append(delta_var)
            future_positions.append(next_pos)
            
            current_pos = next_pos
        
        # Stack predictions
        deltas_mu = torch.stack(future_deltas_mu, dim=1)
        deltas_var = torch.stack(future_deltas_var, dim=1)
        positions = torch.stack(future_positions, dim=1)
        
        # Note: For sampling, we use the actual sampled positions, 
        # but we could also propagate the variance analytically
        positions_var = self._propagate_variance(deltas_var)
        
        return {
            'future_deltas_mu': deltas_mu.view(batch_size, num_agents, self.T_future, 2),
            'future_deltas_var': deltas_var.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_mu': positions.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_var': positions_var.view(batch_size, num_agents, self.T_future, 2)
        }


# Utility functions for evaluation
def compute_multi_agent_metrics(predictions, targets, video_stats=None):
    """Compute evaluation metrics for multi-agent trajectories"""
    metrics = {}
    
    # Get predictions and targets
    if 'future_positions_mu' in predictions:
        pred_pos = predictions['future_positions_mu']  # (B, N, T_future, 2)
        pred_var = predictions['future_positions_var']  # (B, N, T_future, 2)
    else:
        pred_pos = predictions['future_positions']
        pred_var = None
    
    target_pos = targets['future_positions']  # (B, N, T_future, 2)
    valid_mask = targets.get('occ_mask', torch.ones_like(target_pos[..., 0]))  # (B, N, T_future)
    
    # Denormalize if needed
    if video_stats is not None:
        pred_pos = denormalize_positions(pred_pos, video_stats)
        target_pos = denormalize_positions(target_pos, video_stats)
    
    # Compute ADE (Average Displacement Error) across all agents
    displacement = torch.norm(pred_pos - target_pos, dim=-1)  # (B, N, T_future)
    masked_displacement = displacement * valid_mask
    ade = masked_displacement.sum() / valid_mask.sum()
    metrics['ADE'] = ade.item()
    
    # Compute FDE (Final Displacement Error) across all agents
    final_displacement = displacement[..., -1]  # (B, N)
    final_valid = valid_mask[..., -1]  # (B, N)
    fde = (final_displacement * final_valid).sum() / final_valid.sum()
    metrics['FDE'] = fde.item()
    
    # Per-agent metrics
    batch_size, num_agents = pred_pos.shape[:2]
    agent_ades = []
    agent_fdes = []
    
    for n in range(num_agents):
        agent_disp = displacement[:, n, :]  # (B, T_future)
        agent_mask = valid_mask[:, n, :]  # (B, T_future)
        
        if agent_mask.sum() > 0:
            agent_ade = (agent_disp * agent_mask).sum() / agent_mask.sum()
            agent_ades.append(agent_ade.item())
            
            agent_final_disp = agent_disp[:, -1]  # (B,)
            agent_final_mask = agent_mask[:, -1]  # (B,)
            if agent_final_mask.sum() > 0:
                agent_fde = (agent_final_disp * agent_final_mask).sum() / agent_final_mask.sum()
                agent_fdes.append(agent_fde.item())
    
    if agent_ades:
        metrics['Agent_ADE_Mean'] = sum(agent_ades) / len(agent_ades)
        metrics['Agent_ADE_Std'] = torch.tensor(agent_ades).std().item()
    
    if agent_fdes:
        metrics['Agent_FDE_Mean'] = sum(agent_fdes) / len(agent_fdes)
        metrics['Agent_FDE_Std'] = torch.tensor(agent_fdes).std().item()
    
    # Uncertainty calibration metrics
    if pred_var is not None:
        # Prediction interval coverage (95% confidence)
        std = torch.sqrt(pred_var)
        lower_bound = pred_pos - 1.96 * std
        upper_bound = pred_pos + 1.96 * std
        
        within_interval = ((target_pos >= lower_bound) & (target_pos <= upper_bound)).float()
        coverage = (within_interval * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum()
        metrics['Coverage_95'] = coverage.item()
        
        # Average uncertainty
        avg_uncertainty = (pred_var * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum()
        metrics['Avg_Uncertainty'] = avg_uncertainty.item()
    
    return metrics

# def denormalize_positions(positions, video_stats):
#     """Denormalize positions using video statistics"""
#     # Assuming video_stats contains normalization parameters
#     # This would need to be implemented based on your specific normalization scheme
#     return positions  # Placeholder - implement based on your normalization

# Example usage and loss function
class MultiAgentTrajectoryLoss(nn.Module):
    def __init__(self, position_weight=1.0, delta_weight=1.0, uncertainty_weight=0.1):
        super().__init__()
        self.position_weight = position_weight
        self.delta_weight = delta_weight
        self.uncertainty_weight = uncertainty_weight
        
    def forward(self, predictions, targets):
        """
        predictions: dict with 'future_positions_mu', 'future_positions_var', 
                     'future_deltas_mu', 'future_deltas_var'
        targets: dict with 'future_positions', 'future_deltas' (if available), 'occ_mask'
        """
        losses = {}
        total_loss = 0
        
        # Get masks
        valid_mask = targets.get('occ_mask', torch.ones_like(targets['future_positions'][..., 0]))
        
        # Position loss (negative log-likelihood of Gaussian)
        if 'future_positions_mu' in predictions:
            pos_mu = predictions['future_positions_mu']
            pos_var = predictions['future_positions_var']
            target_pos = targets['future_positions']
            
            # NLL loss: 0.5 * (log(2*pi*var) + (x-mu)^2/var)
            pos_diff = target_pos - pos_mu  # (B, N, T, 2)
            pos_nll = 0.5 * (torch.log(2 * math.pi * pos_var) + pos_diff**2 / pos_var)
            pos_nll = pos_nll.sum(dim=-1)  # Sum over x,y coordinates
            
            # Apply mask and average
            pos_loss = (pos_nll * valid_mask).sum() / valid_mask.sum()
            losses['position_loss'] = pos_loss
            total_loss += self.position_weight * pos_loss
        
        # Delta loss (if available)
        if 'future_deltas_mu' in predictions and 'future_deltas' in targets:
            delta_mu = predictions['future_deltas_mu']
            delta_var = predictions['future_deltas_var']
            target_delta = targets['future_deltas']
            
            delta_diff = target_delta - delta_mu
            delta_nll = 0.5 * (torch.log(2 * math.pi * delta_var) + delta_diff**2 / delta_var)
            delta_nll = delta_nll.sum(dim=-1)
            
            delta_loss = (delta_nll * valid_mask).sum() / valid_mask.sum()
            losses['delta_loss'] = delta_loss
            total_loss += self.delta_weight * delta_loss
        
        # Regularization on uncertainty (prevent it from becoming too large)
        if 'future_positions_var' in predictions:
            var_reg = predictions['future_positions_var'].mean()
            losses['uncertainty_reg'] = var_reg
            total_loss += self.uncertainty_weight * var_reg
        
        losses['total_loss'] = total_loss
        return losses
