# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
        
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x: (B, T, d_model) for batch_first=True
#         batch_size, seq_len = x.size(0), x.size(1)
#         # pe: (max_len, d_model) -> (1, seq_len, d_model) -> (B, seq_len, d_model)
#         pe_slice = self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
#         return x + pe_slice

# class SpatialEncoding(nn.Module):
#     def __init__(self, d_model, max_pos=10000):
#         super(SpatialEncoding, self).__init__()
#         self.d_model = d_model
#         self.max_pos = max_pos
#         self.half_dim = d_model//2

#         div_term = torch.exp(
#             torch.arange(0, self.half_dim, 2).float() *
#             (-math.log(10000.0) / self.half_dim)
#         )
#         self.register_buffer('div_term', div_term)

#     def forward(self, positions: torch.Tensor) -> torch.Tensor:
#         """
#         positions: (..., 2) — normalized x,y in [0,1]
#         returns: (..., d_model)
#         """
#         original_shape = positions.shape
#         positions_flat = positions.view(-1, 2)
#         batch_flat_size = positions_flat.size(0)
        
#         scaled = positions_flat * self.max_pos
        
#         pe_x = torch.zeros(batch_flat_size, self.half_dim, device=positions.device)
#         pe_x[:, 0::2] = torch.sin(scaled[:, 0:1] * self.div_term.unsqueeze(0))
#         pe_x[:, 1::2] = torch.cos(scaled[:, 0:1] * self.div_term.unsqueeze(0))

#         pe_y = torch.zeros(batch_flat_size, self.half_dim, device=positions.device)
#         pe_y[:, 0::2] = torch.sin(scaled[:, 1:2] * self.div_term.unsqueeze(0))
#         pe_y[:, 1::2] = torch.cos(scaled[:, 1:2] * self.div_term.unsqueeze(0))
        
#         pe = torch.cat([pe_x, pe_y], dim=-1)
#         return pe.view(*original_shape[:-1], self.d_model)



# class AlternatingTransformerEncoder(nn.Module):
#     """Encoder that alternates between temporal and agent attention with proper masking"""
#     def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
#         super(AlternatingTransformerEncoder, self).__init__()
#         self.layers = nn.ModuleList([
#             self._create_encoder_layer(d_model, nhead, dim_feedforward, dropout)
#             for _ in range(num_layers)
#         ])
#         self.num_layers = num_layers
        
#     def _create_encoder_layer(self, d_model, nhead, dim_feedforward, dropout):
#         return nn.ModuleDict({
#             'temporal': nn.TransformerEncoderLayer(
#                 d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
#             ),
#             'agent': nn.TransformerEncoderLayer(
#                 d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
#             )
#         })
    
#     def _create_temporal_mask(self, obs_masks, agent_masks, batch_size, num_agents, seq_len):
#         """Create mask for temporal attention (B*N, T)"""
#         # obs_masks: (B, N, T) - False where observed, True where padded
#         # agent_masks: (B, N) - 1 for existing agents, 0 for non-existent
        
#         obs_mask_flat = obs_masks.view(batch_size * num_agents, seq_len)  # (B*N, T)
#         agent_mask_expanded = agent_masks.unsqueeze(-1).expand(-1, -1, seq_len)  # (B, N, T)
#         agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * num_agents, seq_len)  # (B*N, T)
        
#         # For temporal attention: mask padding AND ensure non-existent agents have at least one unmasked position
#         # to prevent complete sequence masking
#         temporal_mask = (obs_mask_flat == 0)  # True where padded (inverse of obs_mask)
        
#         # For non-existent agents, create a special mask that leaves first position unmasked
#         # This prevents NaN while still allowing the model to learn to ignore these agents
#         non_existent_agents = (agent_mask_flat == 0)
#         special_mask = temporal_mask.clone()
#         special_mask[non_existent_agents.all(dim=1), 0] = False  # Unmask first position for non-existent agents
        
#         return special_mask
    
#     def _create_agent_mask(self, obs_masks, agent_masks, batch_size, num_agents, seq_len):
#         """Create mask for agent attention (B*T, N)"""
#         # For agent attention, we need to consider which agents exist at each timestep
        
#         # Agent existence mask - this MUST be strict for agent attention
#         agent_mask_expanded = agent_masks.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, N)
#         agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * seq_len, num_agents)  # (B*T, N)
        
#         # For agent attention: STRICTLY mask non-existent agents
#         # This is critical - non-existent agents MUST NOT influence existing agents
#         return (agent_mask_flat == 0)  # True where agent doesn't exist
    
#     def forward(self, x, batch_size, num_agents, seq_len, obs_masks=None, agent_masks=None):
#         """
#         x: (B*N, T, d_model) - input format
#         obs_masks: (B, N, T) - 1 where observed, 0 where padded
#         agent_masks: (B, N) - 1 where agent exists, 0 where non-existent
#         Returns: (B*N, T, d_model)
#         """
#         current_x = x
        
#         # Create masks once
#         temporal_mask = None
#         agent_mask = None
        
#         if obs_masks is not None and agent_masks is not None:
#             temporal_mask = self._create_temporal_mask(obs_masks, agent_masks, batch_size, num_agents, seq_len)
#             agent_mask = self._create_agent_mask(obs_masks, agent_masks, batch_size, num_agents, seq_len)
        
#         for i, layer_dict in enumerate(self.layers):
#             if i % 2 == 0:
#                 # Even layer: temporal attention (B*N, T, d_model)
#                 layer = layer_dict['temporal']
#                 current_x = self._ensure_temporal_format(current_x, batch_size, num_agents, seq_len)
#                 current_x = layer(current_x, src_key_padding_mask=temporal_mask)
                
#                 # CRITICAL: Zero out non-existent agents after temporal attention
#                 if agent_masks is not None:
#                     current_x = self._zero_out_nonexistent_agents_temporal(
#                         current_x, agent_masks, batch_size, num_agents, seq_len
#                     )
                    
#             else:
#                 # Odd layer: agent attention (B*T, N, d_model)
#                 layer = layer_dict['agent']
#                 current_x = self._ensure_agent_format(current_x, batch_size, num_agents, seq_len)
#                 current_x = layer(current_x, src_key_padding_mask=agent_mask)
                
#                 # CRITICAL: Zero out non-existent agents after agent attention
#                 if agent_masks is not None:
#                     current_x = self._zero_out_nonexistent_agents_agent(
#                         current_x, agent_masks, batch_size, num_agents, seq_len
#                     )
        
#         # Ensure final output is in (B*N, T, d_model) format
#         current_x = self._ensure_temporal_format(current_x, batch_size, num_agents, seq_len)
        
#         # FINAL: Ensure non-existent agents are completely zeroed
#         if agent_masks is not None:
#             current_x = self._zero_out_nonexistent_agents_temporal(
#                 current_x, agent_masks, batch_size, num_agents, seq_len
#             )
        
#         return current_x
    
#     def _zero_out_nonexistent_agents_temporal(self, x, agent_masks, batch_size, num_agents, seq_len):
#         """Zero out non-existent agents in temporal format (B*N, T, d_model)"""
#         # x: (B*N, T, d_model)
#         agent_mask_expanded = agent_masks.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, seq_len, x.size(-1))  # (B, N, T, d_model)
#         agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * num_agents, seq_len, x.size(-1))  # (B*N, T, d_model)
#         return x * agent_mask_flat
    
#     def _zero_out_nonexistent_agents_agent(self, x, agent_masks, batch_size, num_agents, seq_len):
#         """Zero out non-existent agents in agent format (B*T, N, d_model)"""
#         # x: (B*T, N, d_model)
#         agent_mask_expanded = agent_masks.unsqueeze(1).unsqueeze(-1).expand(-1, seq_len, -1, x.size(-1))  # (B, T, N, d_model)
#         agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * seq_len, num_agents, x.size(-1))  # (B*T, N, d_model)
#         return x * agent_mask_flat




# # class AlternatingTransformerEncoder(nn.Module):
# #     """Encoder that alternates between temporal and agent attention"""
# #     def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
# #         super(AlternatingTransformerEncoder, self).__init__()
# #         self.layers = nn.ModuleList([
# #             self._create_encoder_layer(d_model, nhead, dim_feedforward, dropout)
# #             for _ in range(num_layers)
# #         ])
# #         self.num_layers = num_layers
        
# #     def _create_encoder_layer(self, d_model, nhead, dim_feedforward, dropout):
# #         return nn.ModuleDict({
# #             'temporal': nn.TransformerEncoderLayer(
# #                 d_model, nhead, dim_feedforward, dropout, batch_first=True
# #             ),
# #             'agent': nn.TransformerEncoderLayer(
# #                 d_model, nhead, dim_feedforward, dropout, batch_first=True
# #             )
# #         })
    
# #     def _ensure_temporal_format(self, tensor, batch_size, num_agents, seq_len):
# #         """Ensure tensor is in (B*N, T, d_model) format"""
# #         if tensor.shape[0] == batch_size * num_agents and tensor.shape[1] == seq_len:
# #             # Already in correct format
# #             return tensor
# #         elif tensor.shape[0] == batch_size * seq_len and tensor.shape[1] == num_agents:
# #             # Convert from (B*T, N, d_model) to (B*N, T, d_model)
# #             d_model = tensor.shape[2]
# #             tensor = tensor.view(batch_size, seq_len, num_agents, d_model)
# #             tensor = tensor.permute(0, 2, 1, 3).contiguous()
# #             tensor = tensor.view(batch_size * num_agents, seq_len, d_model)
# #             return tensor
# #         else:
# #             raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
# #     def _ensure_agent_format(self, tensor, batch_size, num_agents, seq_len):
# #         """Ensure tensor is in (B*T, N, d_model) format"""
# #         if tensor.shape[0] == batch_size * seq_len and tensor.shape[1] == num_agents:
# #             # Already in correct format
# #             return tensor
# #         elif tensor.shape[0] == batch_size * num_agents and tensor.shape[1] == seq_len:
# #             # Convert from (B*N, T, d_model) to (B*T, N, d_model)
# #             d_model = tensor.shape[2]
# #             tensor = tensor.view(batch_size, num_agents, seq_len, d_model)
# #             tensor = tensor.permute(0, 2, 1, 3).contiguous()
# #             tensor = tensor.view(batch_size * seq_len, num_agents, d_model)
# #             return tensor
# #         else:
# #             raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
# #     def _reshape_mask_for_agent_attention(self, mask, batch_size, num_agents, seq_len):
# #         """Reshape mask from (B*N, T) to (B*T, N) for agent attention"""
# #         if mask is None:
# #             return None
# #         # mask: (B*N, T) -> (B, N, T) -> (B, T, N) -> (B*T, N)
# #         mask = mask.view(batch_size, num_agents, seq_len)
# #         mask = mask.permute(0, 2, 1).contiguous()
# #         mask = mask.view(batch_size * seq_len, num_agents)
# #         return mask
    
# #     def _reshape_mask_for_temporal_attention(self, mask, batch_size, num_agents, seq_len):
# #         """Reshape mask from (B*T, N) to (B*N, T) for temporal attention"""
# #         if mask is None:
# #             return None
# #         # mask: (B*T, N) -> (B, T, N) -> (B, N, T) -> (B*N, T)
# #         mask = mask.view(batch_size, seq_len, num_agents)
# #         mask = mask.permute(0, 2, 1).contiguous()
# #         mask = mask.view(batch_size * num_agents, seq_len)
# #         return mask
        
# #     def forward(self, x, batch_size, num_agents, seq_len, attention_mask=None):
# #         """
# #         x: (B*N, T, d_model) - input format
# #         attention_mask: (B*N, T) - padding mask
# #         Returns: (B*N, T, d_model)
# #         """
# #         current_x = x
# #         current_mask = attention_mask
        
# #         for i, layer_dict in enumerate(self.layers):
# #             if i % 2 == 0:
# #                 # Even layer: temporal attention (B*N, T, d_model)
# #                 layer = layer_dict['temporal']
                
# #                 # Ensure correct format for temporal attention
# #                 current_x = self._ensure_temporal_format(current_x, batch_size, num_agents, seq_len)
                
# #                 # Make sure mask is in temporal format (B*N, T)
# #                 if current_mask is not None and current_mask.shape != (batch_size * num_agents, seq_len):
# #                     current_mask = self._reshape_mask_for_temporal_attention(
# #                         current_mask, batch_size, num_agents, seq_len
# #                     )
                    
# #             else:
# #                 # Odd layer: agent attention (B*T, N, d_model)
# #                 layer = layer_dict['agent']
                
# #                 # Ensure correct format for agent attention
# #                 current_x = self._ensure_agent_format(current_x, batch_size, num_agents, seq_len)
                
# #                 # Reshape mask for agent attention (B*T, N)
# #                 if current_mask is not None:
# #                     current_mask = self._reshape_mask_for_agent_attention(
# #                         current_mask, batch_size, num_agents, seq_len
# #                     )
            
# #             # Apply the layer
# #             current_x = layer(current_x, src_key_padding_mask=current_mask)
        
# #         # Ensure final output is in (B*N, T, d_model) format
# #         current_x = self._ensure_temporal_format(current_x, batch_size, num_agents, seq_len)
        
# #         return current_x

# class StandardTransformerDecoder(nn.Module):
#     """Standard decoder that operates on (B*N, T, d_model) throughout"""
#     def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
#         super(StandardTransformerDecoder, self).__init__()
#         self.layers = nn.ModuleList([
#             nn.TransformerDecoderLayer(
#                 d_model, nhead, dim_feedforward, dropout, batch_first=True
#             )
#             for _ in range(num_layers)
#         ])
#         self.num_layers = num_layers
        
#     def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
#         """
#         tgt: (B*N, T_future, d_model)
#         memory: (B*N, T_past, d_model)
#         Returns: (B*N, T_future, d_model)
#         """
#         current_tgt = tgt
        
#         for layer in self.layers:
#             current_tgt = layer(
#                 tgt=current_tgt,
#                 memory=memory,
#                 tgt_mask=tgt_mask,
#                 memory_key_padding_mask=memory_key_padding_mask
#             )
        
#         return current_tgt
    
# class ImprovedGraphInteractionModel(nn.Module):
#     def __init__(self, num_classes, locations, d_model = 256, nhead = 8, num_layers = 6, 
#                  T_past = 10, T_future = 10, max_agents = 50):
#         super().__init__()
#         self.num_classes = num_classes
#         self.locations = locations
#         self.d_model = d_model
#         self.nhead = nhead
#         self.num_layers = num_layers
#         self.T_past = T_past
#         self.T_future = T_future
#         self.max_agents = 50

#         self.location_to_idx = {loc:i for i,loc in enumerate(locations)}

#         self.class_embedding = nn.Embedding(num_classes, d_model)
#         self.location_embedding = nn.Embedding(len(locations), d_model)

#         self.spatial_encoding = SpatialEncoding(d_model)
#         self.temporal_encoding = PositionalEncoding(d_model)

#         self.pos_projection = nn.Sequential(
#             nn.Linear(2, self.d_model),
#             nn.LayerNorm(d_model)
#         )

#         self.unobserved_token = nn.Parameter(torch.randn(self.d_model) * 0.1)
#         self.gap_encoding = nn.Embedding(T_past + 1, d_model)

#         self.transformer_encoder = AlternatingTransformerEncoder(
#             self.d_model, self.nhead, self.num_layers, 4 * self.d_model
#         )

#         self.transformer_decoder = StandardTransformerDecoder(
#             self.d_model, self.nhead, self.num_layers, 4 * self.d_model
#         )

#         self.output_norm = nn.LayerNorm(d_model)

#         self.delta_projection = nn.Sequential(
#             nn.Linear(d_model, d_model // 2),
#             nn.LayerNorm(d_model // 2),
#             nn.ReLU(),
#             nn.Linear(d_model // 2, 2)
#         )

#         self.uncertainty_projection = nn.Sequential(
#             nn.Linear(d_model, d_model // 4),
#             nn.LayerNorm(d_model // 4),
#             nn.ReLU(),
#             nn.Linear(d_model // 4, 2)
#         )

#         self.position_interpolator = nn.Sequential(
#             nn.Linear(d_model, d_model // 2),
#             nn.LayerNorm(d_model // 2),
#             nn.ReLU(),
#             nn.Linear(d_model // 2, 2)
#         )

#         self._initialize_weights()
        
#         # IMPROVED: Learned parameters for stability (Trajectron++ approach)
#         self.register_parameter('log_sigma_init', nn.Parameter(torch.tensor(-2.0)))
#         self.register_parameter('max_delta_scale', nn.Parameter(torch.tensor(20.0)))


#     def _initialize_weights(self):
#         for module in self.modules():
#             #xavier/glorot init
#             if isinstance(module, nn.Linear): 
#                 nn.init.xavier_uniform_(module.weight, gain = 0.1)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#             elif isinstance(module, nn.Embedding):
#                 nn.init.normal_(module.weight, std = 0.1)

#         nn.init.xavier_uniform_(self.delta_projection[-1].weight, gain = 0.01)
#         nn.init.zeros_(self.delta_projection[-1].bias)

#         nn.init.xavier_uniform_(self.uncertainty_projection[-1].weight, gain = 0.01)
#         nn.init.constant_(self.uncertainty_projection[-1].bias, -2)

#     # def _create_enhanced_past_embeddings(self, past_positions, past_positions_orig, obs_mask,
#     #                                 location_emb, class_emb, batch, device):
#     #     """Create enhanced embeddings with proper agent masking"""
#     #     batch_size, num_agents, seq_len, _ = past_positions.shape

#     #     if torch.isnan(past_positions).any():
#     #         print("WARNING: NaN in past_positions input!")
#     #         past_positions = torch.nan_to_num(past_positions, nan=0.0)
        
#     #     # Calculate gaps correctly (same as before)
#     #     gaps = torch.zeros_like(obs_mask, dtype=torch.long, device=device)
#     #     for b in range(batch_size):
#     #         for n in range(num_agents):
#     #             # Skip non-existent agents
#     #             if batch['agent_masks'][b, n] == 0:
#     #                 continue
                    
#     #             gap_counter = 0
#     #             for t in range(seq_len):
#     #                 if obs_mask[b, n, t] == 1:
#     #                     gap_counter = 0
#     #                 else:
#     #                     gap_counter += 1
#     #                 gaps[b, n, t] = min(gap_counter, self.T_past)
        
#     #     # Rest of the method stays the same, but add agent masking at the end
#     #     pos_emb = torch.zeros(batch_size, num_agents, seq_len, self.d_model, device=device)
#     #     location_expanded = location_emb.unsqueeze(1).unsqueeze(2).expand(-1, num_agents, seq_len, -1)

#     #     # observed_mask = obs_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
#     #     observed_mask = (obs_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model) == 1.0)
        
#     #     observed_pos_emb = self.pos_projection(past_positions)
#     #     unobserved_emb = self.unobserved_token.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
#     #         batch_size, num_agents, seq_len, -1)
        
#     #     pos_emb = torch.where(observed_mask, observed_pos_emb, unobserved_emb)
        
#     #     spatial_emb = self.spatial_encoding(past_positions_orig)
#     #     pos_emb = pos_emb + spatial_emb * observed_mask
        
#     #     gap_emb = self.gap_encoding(gaps)
#     #     pos_emb = pos_emb + gap_emb
        
#     #     # Add context embeddings with agent masking
#     #     class_expanded = class_emb.unsqueeze(2).expand(-1, -1, seq_len, -1)
#     #     agent_mask_expanded = batch['agent_masks'].unsqueeze(2).unsqueeze(3).expand(-1, -1, seq_len, self.d_model)
        
#     #     # Apply agent mask to class embeddings
#     #     class_expanded_masked = class_expanded * agent_mask_expanded
#     #     pos_emb = pos_emb + class_expanded_masked + location_expanded
        
#     #     # Apply agent mask to final embeddings
#     #     pos_emb = pos_emb * agent_mask_expanded
        
#     #     # Reshape for transformer
#     #     pos_emb = pos_emb.view(batch_size * num_agents, seq_len, self.d_model)
#     #     pos_emb = self.temporal_encoding(pos_emb)
        
#     #     return pos_emb

#     def _create_enhanced_past_embeddings(self, past_positions, past_positions_orig, obs_mask,
#                                     location_emb, class_emb, batch, device):
#         """Create enhanced embeddings with proper agent masking"""
#         batch_size, num_agents, seq_len, _ = past_positions.shape

#         if torch.isnan(past_positions).any():
#             print("WARNING: NaN in past_positions input!")
#             past_positions = torch.nan_to_num(past_positions, nan=0.0)
        
#         # Calculate gaps correctly
#         gaps = torch.zeros_like(obs_mask, dtype=torch.long, device=device)
#         for b in range(batch_size):
#             for n in range(num_agents):
#                 if batch['agent_masks'][b, n] == 0:
#                     continue
                    
#                 gap_counter = 0
#                 for t in range(seq_len):
#                     if obs_mask[b, n, t] == 1:
#                         gap_counter = 0
#                     else:
#                         gap_counter += 1
#                     gaps[b, n, t] = min(gap_counter, self.T_past)
        
#         # Create embeddings
#         pos_emb = torch.zeros(batch_size, num_agents, seq_len, self.d_model, device=device)
#         location_expanded = location_emb.unsqueeze(1).unsqueeze(2).expand(-1, num_agents, seq_len, -1)

#         observed_mask = (obs_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model) == 1.0)
        
#         # Use small epsilon instead of exact zeros for unobserved positions to prevent NaN
#         observed_pos_emb = self.pos_projection(past_positions)
#         #IMP: I don't think it's needed, since unobserved tokens aren't 0.
#         # unobserved_emb = self.unobserved_token.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
#         #     batch_size, num_agents, seq_len, -1) + torch.randn_like(self.unobserved_token) * 1e-6

#         unobserved_emb = self.unobserved_token.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
#             batch_size, num_agents, seq_len, -1) + torch.randn_like(self.unobserved_token) * 1e-6
        
#         pos_emb = torch.where(observed_mask, observed_pos_emb, unobserved_emb)
        
#         # Add spatial encoding with numerical stability
#         spatial_emb = self.spatial_encoding(past_positions_orig)
#         spatial_emb = torch.nan_to_num(spatial_emb, nan=0.0)  # Handle any NaN from spatial encoding
#         pos_emb = pos_emb + spatial_emb * observed_mask
        
#         gap_emb = self.gap_encoding(gaps)
#         pos_emb = pos_emb + gap_emb
        
#         # Add context embeddings with agent masking
#         class_expanded = class_emb.unsqueeze(2).expand(-1, -1, seq_len, -1)
#         agent_mask_expanded = batch['agent_masks'].unsqueeze(2).unsqueeze(3).expand(-1, -1, seq_len, self.d_model)
        
#         # Apply agent mask to class embeddings
#         class_expanded_masked = class_expanded * agent_mask_expanded
#         pos_emb = pos_emb + class_expanded_masked + location_expanded
        
#         # Apply agent mask to final embeddings - but add small epsilon to prevent all-zero embeddings
#         epsilon_embedding = torch.randn_like(pos_emb) * 1e-7
#         pos_emb = pos_emb * agent_mask_expanded + epsilon_embedding * (1 - agent_mask_expanded)
        
#         # Reshape for transformer
#         pos_emb = pos_emb.view(batch_size * num_agents, seq_len, self.d_model)
#         pos_emb = self.temporal_encoding(pos_emb)
        
#         # Final NaN check
#         if torch.isnan(pos_emb).any():
#             print("WARNING: NaN in final embeddings!")
#             pos_emb = torch.nan_to_num(pos_emb, nan=0.0)
        
#         return pos_emb

#     def _teacher_forcing_predictions(self, memory, attention_mask, current_pos_est,
#                                future_positions_target, location_emb, class_emb,
#                                batch_size, num_agents, device, agent_masks=None):
#         """IMPROVED: Following Trajectron++ approach with soft constraints"""
#         current_pos_flat = current_pos_est.view(batch_size * num_agents, 2)
#         future_pos_flat = future_positions_target.view(batch_size * num_agents, self.T_future, 2)

#         decoder_positions = torch.cat([
#             current_pos_flat.unsqueeze(1),
#             future_pos_flat[:, :-1, :]
#         ], dim=1)
#         decoder_input = self.pos_projection(decoder_positions)

#         location_flat = location_emb.unsqueeze(1).expand(batch_size, num_agents, -1).contiguous()
#         location_flat = location_flat.view(batch_size * num_agents, self.d_model)
#         class_flat = class_emb.view(batch_size * num_agents, self.d_model)

#         context_emb = class_flat + location_flat
#         decoder_input = decoder_input + context_emb.unsqueeze(1)

#         if agent_masks is not None:
#             agent_mask_flat = agent_masks.view(batch_size * num_agents, 1, 1)
#             decoder_input = decoder_input * agent_mask_flat

#         future_temporal_emb = torch.zeros(self.T_future, self.d_model, device=device)
#         max_pe_len = self.temporal_encoding.pe.size(0)

#         for i in range(self.T_future):
#             temporal_pos = self.T_past + i
#             if temporal_pos < max_pe_len:
#                 future_temporal_emb[i] = self.temporal_encoding.pe[temporal_pos]

#         decoder_input = decoder_input + future_temporal_emb.unsqueeze(0).expand(batch_size * num_agents, -1, -1)

#         causal_mask = torch.triu(
#             torch.ones(self.T_future, self.T_future, device=device) * float('-inf'), 
#             diagonal=1
#         )

#         decoder_output = self.transformer_decoder(tgt = decoder_input, memory = memory, tgt_mask = causal_mask,
#                                                   memory_key_padding_mask=attention_mask)
#         # IMPROVED: Layer norm before output projections
#         normalized_output = self.output_norm(decoder_output)

#         # IMPROVED: Soft-constrained outputs using tanh scaling (Social-GAN style)
#         delta_raw = self.delta_projection(normalized_output)
#         delta_mu = torch.tanh(delta_raw) * self.max_delta_scale

#         # IMPROVED: Conservative uncertainty prediction
#         log_var_raw = self.uncertainty_projection(normalized_output)
#         log_var = self.log_sigma_init + torch.tanh(log_var_raw) * 2.0  # Range: [log_sigma_init-2, log_sigma_init+2]

#         # Convert deltas to positions
#         positions_mu = self._convert_deltas_to_positions_teacher_forcing(
#             current_pos_flat, delta_mu
#         )

#         # IMPROVED: Proper variance propagation without explosion
#         delta_var = torch.exp(log_var)
#         positions_var = self._propagate_variance_proper(delta_var)

#         return {
#             'future_deltas_mu': delta_mu.view(batch_size, num_agents, self.T_future, 2),
#             'future_deltas_logvar': log_var.view(batch_size, num_agents, self.T_future, 2),
#             'future_positions_mu': positions_mu.view(batch_size, num_agents, self.T_future, 2),
#             'future_positions_var': positions_var.view(batch_size, num_agents, self.T_future, 2)
#         }
    
#     def _propagate_variance_proper(self, delta_var):
#         """Proper variance propagation following probability theory"""
#         # For independent Gaussians: Var(X + Y) = Var(X) + Var(Y)
#         pos_var = torch.cumsum(delta_var, dim=1)  # Much cleaner!
#         return pos_var
    

#     def _generate_autoregressive_predictions(self, memory, attention_mask, current_pos_est,
#                                            class_emb, location_emb, 
#                                            batch_size, num_agents, device):
#         """IMPROVED: Autoregressive generation with proper constraints"""
        
#         current_pos = current_pos_est.view(batch_size * num_agents, 2)
        
#         # Context embeddings
#         class_flat = class_emb.view(batch_size * num_agents, self.d_model)
#         location_flat = location_emb.unsqueeze(1).expand(batch_size, num_agents, -1).contiguous()
#         location_flat = location_flat.view(batch_size * num_agents, self.d_model)
#         context_emb = class_flat + location_flat
        
#         # Storage for predictions
#         future_deltas_mu = []
#         future_deltas_logvar = []
#         future_positions = []
        
#         for t in range(self.T_future):
#             # Create decoder input
#             decoder_input = self.pos_projection(current_pos).unsqueeze(1)
#             decoder_input = decoder_input + context_emb.unsqueeze(1)
            
#             # Add temporal encoding
#             temporal_pos = self.T_past + t
#             if temporal_pos < self.temporal_encoding.pe.size(0):
#                 temporal_emb = self.temporal_encoding.pe[temporal_pos]
#                 decoder_input = decoder_input + temporal_emb.unsqueeze(0).unsqueeze(0)
            
#             # Apply decoder
#             decoder_output = self.transformer_decoder(
#                 tgt=decoder_input,
#                 memory=memory,
#                 tgt_mask=None,
#                 memory_key_padding_mask=attention_mask
#             )
            
#             # IMPROVED: Same soft constraints as teacher forcing
#             normalized_output = self.output_norm(decoder_output.squeeze(1))
            
#             delta_raw = self.delta_projection(normalized_output)
#             delta_mu = torch.tanh(delta_raw) * self.max_delta_scale
            
#             log_var_raw = self.uncertainty_projection(normalized_output)
#             log_var = self.log_sigma_init + torch.tanh(log_var_raw) * 2.0
            
#             # Update position
#             next_pos = current_pos + delta_mu
            
#             # Store predictions
#             future_deltas_mu.append(delta_mu)
#             future_deltas_logvar.append(log_var)
#             future_positions.append(next_pos)
            
#             current_pos = next_pos
        
#         # Stack predictions
#         deltas_mu = torch.stack(future_deltas_mu, dim=1)
#         deltas_logvar = torch.stack(future_deltas_logvar, dim=1)
#         positions = torch.stack(future_positions, dim=1)
        
#         # Proper variance propagation
#         delta_var = torch.exp(deltas_logvar)
#         positions_var = self._propagate_variance_proper(delta_var)
        
#         return {
#             'future_deltas_mu': deltas_mu.view(batch_size, num_agents, self.T_future, 2),
#             'future_deltas_logvar': deltas_logvar.view(batch_size, num_agents, self.T_future, 2),
#             'future_positions_mu': positions.view(batch_size, num_agents, self.T_future, 2),
#             'future_positions_var': positions_var.view(batch_size, num_agents, self.T_future, 2)
#         }
    
#     def _convert_deltas_to_positions_teacher_forcing(self, start_pos, deltas):
#         """Convert delta predictions to positions during teacher forcing"""
#         positions = torch.zeros_like(deltas)  # (B*N, T_future, 2)
#         current_pos = start_pos  # (B*N, 2)
        
#         for i in range(deltas.size(1)):
#             current_pos = current_pos + deltas[:, i, :]
#             positions[:, i, :] = current_pos
        
#         return positions
    
#     def _estimate_current_position(self, memory, obs_mask, past_positions, agent_masks=None):
#         """Estimate current position for each agent"""
#         batch_size, num_agents, seq_len, _ = past_positions.shape
        
#         # Reshape memory: (B*N, T, d_model) -> (B, N, T, d_model)
#         memory_reshaped = memory.view(batch_size, num_agents, seq_len, self.d_model)
        
#         # Create attention weights
#         weights = torch.softmax(memory_reshaped.mean(dim=-1), dim=-1)  # (B, N, T)
#         weights = weights * obs_mask
#         weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
#         # Weighted combination
#         context = torch.sum(memory_reshaped * weights.unsqueeze(-1), dim=2)  # (B, N, d_model)
#         current_pos = self.position_interpolator(context)  # (B, N, 2)
        
#         # Zero out positions for non-existent agents if agent_masks provided
#         if agent_masks is not None:
#             current_pos = current_pos * agent_masks.unsqueeze(-1)
        
#         return current_pos
    
#     # def forward(self, batch, use_teacher_forcing=True):
#     #     """Main forward pass with proper non-existent agent handling"""
#     #     batch_size, num_agents = batch['past_positions'].shape[:2]
#     #     device = batch['past_positions'].device
#     #     location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], 
#     #                                 device=device)
        
#     #     # Get embeddings - SAFE VERSION for non-existent agents
#     #     class_emb = self._get_safe_embeddings(batch['agent_labels'], batch['agent_masks'])
#     #     location_emb = self.location_embedding(location_indices)
        
#     #     # Create past embeddings with agent masking
#     #     past_emb = self._create_enhanced_past_embeddings(
#     #         batch['past_positions'], batch['past_positions_orig'], batch['obs_masks'], 
#     #         location_emb, class_emb, batch, device
#     #     )
        
#     #     # Create attention mask - FIXED VERSION
#     #     obs_mask_flat = batch['obs_masks'].view(batch_size * num_agents, self.T_past)
        
#     #     # Expand agent masks properly
#     #     agent_mask_expanded = batch['agent_masks'].unsqueeze(-1).expand(-1, -1, self.T_past)
#     #     agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * num_agents, self.T_past)
        
#     #     # Combine masks: True where padded OR agent doesn't exist
#     #     attention_mask = (obs_mask_flat == 0) | (agent_mask_flat == 0)
        
#     #     # Encode - alternating between temporal and agent attention
#     #     memory = self.transformer_encoder(
#     #         past_emb, 
#     #         batch_size=batch_size, 
#     #         num_agents=num_agents, 
#     #         seq_len=self.T_past,
#     #         attention_mask=attention_mask
#     #     )
        
#     #     # Estimate current positions (only for existing agents)
#     #     current_pos_est = self._estimate_current_position(
#     #         memory, batch['obs_masks'], batch['past_positions'], batch['agent_masks']
#     #     )
        
#     #     # Generate future predictions
#     #     if use_teacher_forcing and self.training:
#     #         predictions = self._teacher_forcing_predictions(
#     #             memory, attention_mask, current_pos_est, batch['future_positions'],
#     #             location_emb, class_emb, batch_size, num_agents, device
#     #         )
#     #     else:
#     #         predictions = self._generate_autoregressive_predictions(
#     #             memory, attention_mask, current_pos_est, class_emb, 
#     #             location_emb, batch_size, num_agents, device
#     #         )
        
#     #     predictions['current_position_estimate'] = current_pos_est
#     #     return predictions

#     def forward(self, batch, use_teacher_forcing=True):
#         """Main forward pass with proper non-existent agent handling"""
#         batch_size, num_agents = batch['past_positions'].shape[:2]
#         device = batch['past_positions'].device
        
#         # Add input validation
#         for key, tensor in batch.items():
#             if torch.isnan(tensor).any():
#                 print(f"WARNING: NaN detected in input {key}")
#                 batch[key] = torch.nan_to_num(tensor, nan=0.0)
        
#         location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], 
#                                     device=device)
        
#         # Get embeddings
#         class_emb = self._get_safe_embeddings(batch['agent_labels'], batch['agent_masks'])
#         location_emb = self.location_embedding(location_indices)
        
#         # Create past embeddings with agent masking
#         past_emb = self._create_enhanced_past_embeddings(
#             batch['past_positions'], batch['past_positions_orig'], batch['obs_masks'], 
#             location_emb, class_emb, batch, device
#         )
        
#         # Check for NaN after embeddings
#         if torch.isnan(past_emb).any():
#             print("WARNING: NaN in past embeddings!")
#             return None
        
#         # Encode with proper masking
#         memory = self.transformer_encoder(
#             past_emb, 
#             batch_size=batch_size, 
#             num_agents=num_agents, 
#             seq_len=self.T_past,
#             obs_masks=batch['obs_masks'],  # Pass original masks
#             agent_masks=batch['agent_masks']
#         )
        
#         # Check for NaN after encoding
#         if torch.isnan(memory).any():
#             print("WARNING: NaN in encoder output!")
#             return None
        
#         # Rest of forward pass...
#         current_pos_est = self._estimate_current_position(
#             memory, batch['obs_masks'], batch['past_positions'], batch['agent_masks']
#         )
        
#         if use_teacher_forcing and self.training:
#             predictions = self._teacher_forcing_predictions(
#                 memory, None, current_pos_est, batch['future_positions'],  # Set attention_mask to None for now
#                 location_emb, class_emb, batch_size, num_agents, device
#             )
#         else:
#             predictions = self._generate_autoregressive_predictions(
#                 memory, None, current_pos_est, class_emb, 
#                 location_emb, batch_size, num_agents, device
#             )
        
#         predictions['current_position_estimate'] = current_pos_est
#         return predictions

#     def _get_safe_embeddings(self, agent_labels, agent_masks):
#         """Get embeddings only for existing agents"""
#         existing_mask = (agent_labels != -1)
#         safe_labels = torch.where(existing_mask, agent_labels, 0)
#         class_emb_all = self.class_embedding(safe_labels)

#         if torch.isnan(class_emb_all).any():
#             print("WARNING: NaN in class embeddings!")
#             class_emb_all = torch.nan_to_num(class_emb_all, nan=0.0)
        
#         # Zero out embeddings for non-existent agents using agent_masks
#         class_emb = class_emb_all * agent_masks.unsqueeze(-1).float()
#         return class_emb
#     def denormalize_positions(normalized_coords, video_stats):
#         """Convert normalized coordinates back to original scale"""
#         if video_stats is None:
#             return normalized_coords
        
#         if isinstance(normalized_coords, torch.Tensor):
#             device = normalized_coords.device
#             mean = torch.tensor(video_stats['mean']).to(device)
#             std = torch.tensor(video_stats['std']).to(device)
#             return normalized_coords * std + mean
#         else:
#             return normalized_coords * video_stats['std'] + video_stats['mean']


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        # x: (B, T, d_model) for batch_first=True
        batch_size, seq_len = x.size(0), x.size(1)
        # pe: (max_len, d_model) -> (1, seq_len, d_model) -> (B, seq_len, d_model)
        pe_slice = self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        return x + pe_slice

class SafeSpatialEncoding(nn.Module):
    """FIXED: Spatial encoding that properly handles padding values"""
    def __init__(self, d_model, max_pos=10000, pad_value=-999.0):
        super(SafeSpatialEncoding, self).__init__()
        self.d_model = d_model
        self.max_pos = max_pos
        self.half_dim = d_model // 2
        self.pad_value = pad_value

        div_term = torch.exp(
            torch.arange(0, self.half_dim, 2).float() *
            (-math.log(10000.0) / self.half_dim)
        )
        self.register_buffer('div_term', div_term)
        
        # Create a learnable "pad embedding" for padded positions
        self.pad_embedding = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: (..., 2) — normalized x,y coordinates or pad_value for padding
        returns: (..., d_model)
        """
        original_shape = positions.shape
        positions_flat = positions.view(-1, 2)
        batch_flat_size = positions_flat.size(0)
        
        # FIXED: Create mask for valid (non-padded) positions
        # Check if either x or y coordinate is the pad value
        valid_mask = ~((positions_flat == self.pad_value).any(dim=1))
        
        # Initialize output with pad embeddings
        pe = self.pad_embedding.unsqueeze(0).expand(batch_flat_size, -1).clone()
        
        if valid_mask.any():
            # Only process valid positions
            valid_positions = positions_flat[valid_mask]
            
            # FIXED: Clamp valid positions to reasonable range to prevent overflow
            # Assume normalized positions should be roughly in [-5, 5] range
            valid_positions = torch.clamp(valid_positions, min=-5.0, max=5.0)
            
            scaled = valid_positions * self.max_pos
            
            # Compute encodings for valid positions only
            valid_count = valid_positions.size(0)
            pe_x = torch.zeros(valid_count, self.half_dim, device=positions.device)
            pe_x[:, 0::2] = torch.sin(scaled[:, 0:1] * self.div_term.unsqueeze(0))
            pe_x[:, 1::2] = torch.cos(scaled[:, 0:1] * self.div_term.unsqueeze(0))

            pe_y = torch.zeros(valid_count, self.half_dim, device=positions.device)
            pe_y[:, 0::2] = torch.sin(scaled[:, 1:2] * self.div_term.unsqueeze(0))
            pe_y[:, 1::2] = torch.cos(scaled[:, 1:2] * self.div_term.unsqueeze(0))
            
            pe_valid = torch.cat([pe_x, pe_y], dim=-1)
            
            # Place valid encodings back into the output tensor
            pe[valid_mask] = pe_valid
        
        return pe.view(*original_shape[:-1], self.d_model)

class AlternatingTransformerEncoder(nn.Module):
    """Encoder that alternates between temporal and agent attention with proper masking"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(AlternatingTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            self._create_encoder_layer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
    def _create_encoder_layer(self, d_model, nhead, dim_feedforward, dropout):
        return nn.ModuleDict({
            'temporal': nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
            ),
            'agent': nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
            )
        })
    
    def _create_temporal_mask(self, obs_masks, agent_masks, batch_size, num_agents, seq_len):
        """Create mask for temporal attention (B*N, T)"""
        # obs_masks: (B, N, T) - False where observed, True where padded
        # agent_masks: (B, N) - 1 for existing agents, 0 for non-existent
        
        obs_mask_flat = obs_masks.view(batch_size * num_agents, seq_len)  # (B*N, T)
        agent_mask_expanded = agent_masks.unsqueeze(-1).expand(-1, -1, seq_len)  # (B, N, T)
        agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * num_agents, seq_len)  # (B*N, T)
        
        # For temporal attention: mask padding AND ensure non-existent agents have at least one unmasked position
        # to prevent complete sequence masking
        temporal_mask = (obs_mask_flat == 0)  # True where padded (inverse of obs_mask)
        
        # For non-existent agents, create a special mask that leaves first position unmasked
        # This prevents NaN while still allowing the model to learn to ignore these agents
        non_existent_agents = (agent_mask_flat == 0)
        special_mask = temporal_mask.clone()
        special_mask[non_existent_agents.all(dim=1), 0] = False  # Unmask first position for non-existent agents
        
        return special_mask
    
    def _create_agent_mask(self, obs_masks, agent_masks, batch_size, num_agents, seq_len):
        """Create mask for agent attention (B*T, N)"""
        # For agent attention, we need to consider which agents exist at each timestep
        
        # Agent existence mask - this MUST be strict for agent attention
        agent_mask_expanded = agent_masks.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, N)
        agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * seq_len, num_agents)  # (B*T, N)
        
        # For agent attention: STRICTLY mask non-existent agents
        # This is critical - non-existent agents MUST NOT influence existing agents
        return (agent_mask_flat == 0)  # True where agent doesn't exist
    
    def forward(self, x, batch_size, num_agents, seq_len, obs_masks=None, agent_masks=None):
        """
        x: (B*N, T, d_model) - input format
        obs_masks: (B, N, T) - 1 where observed, 0 where padded
        agent_masks: (B, N) - 1 where agent exists, 0 where non-existent
        Returns: (B*N, T, d_model)
        """
        current_x = x
        
        # Create masks once
        temporal_mask = None
        agent_mask = None
        
        if obs_masks is not None and agent_masks is not None:
            temporal_mask = self._create_temporal_mask(obs_masks, agent_masks, batch_size, num_agents, seq_len)
            agent_mask = self._create_agent_mask(obs_masks, agent_masks, batch_size, num_agents, seq_len)
        
        for i, layer_dict in enumerate(self.layers):
            if i % 2 == 0:
                # Even layer: temporal attention (B*N, T, d_model)
                layer = layer_dict['temporal']
                current_x = self._ensure_temporal_format(current_x, batch_size, num_agents, seq_len)
                current_x = layer(current_x, src_key_padding_mask=temporal_mask)
                
                # CRITICAL: Zero out non-existent agents after temporal attention
                if agent_masks is not None:
                    current_x = self._zero_out_nonexistent_agents_temporal(
                        current_x, agent_masks, batch_size, num_agents, seq_len
                    )
                    
            else:
                # Odd layer: agent attention (B*T, N, d_model)
                layer = layer_dict['agent']
                current_x = self._ensure_agent_format(current_x, batch_size, num_agents, seq_len)
                current_x = layer(current_x, src_key_padding_mask=agent_mask)
                
                # CRITICAL: Zero out non-existent agents after agent attention
                if agent_masks is not None:
                    current_x = self._zero_out_nonexistent_agents_agent(
                        current_x, agent_masks, batch_size, num_agents, seq_len
                    )
        
        # Ensure final output is in (B*N, T, d_model) format
        current_x = self._ensure_temporal_format(current_x, batch_size, num_agents, seq_len)
        
        # FINAL: Ensure non-existent agents are completely zeroed
        if agent_masks is not None:
            current_x = self._zero_out_nonexistent_agents_temporal(
                current_x, agent_masks, batch_size, num_agents, seq_len
            )
        
        return current_x
    
    def _zero_out_nonexistent_agents_temporal(self, x, agent_masks, batch_size, num_agents, seq_len):
        """Zero out non-existent agents in temporal format (B*N, T, d_model)"""
        # x: (B*N, T, d_model)
        agent_mask_expanded = agent_masks.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, seq_len, x.size(-1))  # (B, N, T, d_model)
        agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * num_agents, seq_len, x.size(-1))  # (B*N, T, d_model)
        return x * agent_mask_flat
    
    def _zero_out_nonexistent_agents_agent(self, x, agent_masks, batch_size, num_agents, seq_len):
        """Zero out non-existent agents in agent format (B*T, N, d_model)"""
        # x: (B*T, N, d_model)
        agent_mask_expanded = agent_masks.unsqueeze(1).unsqueeze(-1).expand(-1, seq_len, -1, x.size(-1))  # (B, T, N, d_model)
        agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * seq_len, num_agents, x.size(-1))  # (B*T, N, d_model)
        return x * agent_mask_flat
    
    def _ensure_temporal_format(self, tensor, batch_size, num_agents, seq_len):
        """Ensure tensor is in (B*N, T, d_model) format"""
        if tensor.shape[0] == batch_size * num_agents and tensor.shape[1] == seq_len:
            # Already in correct format
            return tensor
        elif tensor.shape[0] == batch_size * seq_len and tensor.shape[1] == num_agents:
            # Convert from (B*T, N, d_model) to (B*N, T, d_model)
            d_model = tensor.shape[2]
            tensor = tensor.view(batch_size, seq_len, num_agents, d_model)
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            tensor = tensor.view(batch_size * num_agents, seq_len, d_model)
            return tensor
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    def _ensure_agent_format(self, tensor, batch_size, num_agents, seq_len):
        """Ensure tensor is in (B*T, N, d_model) format"""
        if tensor.shape[0] == batch_size * seq_len and tensor.shape[1] == num_agents:
            # Already in correct format
            return tensor
        elif tensor.shape[0] == batch_size * num_agents and tensor.shape[1] == seq_len:
            # Convert from (B*N, T, d_model) to (B*T, N, d_model)
            d_model = tensor.shape[2]
            tensor = tensor.view(batch_size, num_agents, seq_len, d_model)
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            tensor = tensor.view(batch_size * seq_len, num_agents, d_model)
            return tensor
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

class StandardTransformerDecoder(nn.Module):
    """Standard decoder that operates on (B*N, T, d_model) throughout"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(StandardTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        """
        tgt: (B*N, T_future, d_model)
        memory: (B*N, T_past, d_model)
        Returns: (B*N, T_future, d_model)
        """
        current_tgt = tgt
        
        for layer in self.layers:
            current_tgt = layer(
                tgt=current_tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        return current_tgt

class SafePositionProjection(nn.Module):
    """FIXED: Position projection that handles padding values"""
    def __init__(self, d_model, pad_value=-999.0):
        super(SafePositionProjection, self).__init__()
        self.pad_value = pad_value
        self.projection = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model)
        )
        # Learnable embedding for padded positions
        self.pad_embedding = nn.Parameter(torch.randn(d_model) * 0.01)
    
    def forward(self, positions):
        """
        positions: (..., 2) tensor with potential pad_value entries
        """
        original_shape = positions.shape
        positions_flat = positions.view(-1, 2)
        
        # Create mask for valid positions
        valid_mask = ~((positions_flat == self.pad_value).any(dim=1))
        
        # Initialize output with pad embeddings
        output = self.pad_embedding.unsqueeze(0).expand(positions_flat.size(0), -1).clone()
        
        if valid_mask.any():
            # Process only valid positions
            valid_positions = positions_flat[valid_mask]
            # Clamp to prevent extreme values
            valid_positions = torch.clamp(valid_positions, min=-10.0, max=10.0)
            valid_output = self.projection(valid_positions)
            output[valid_mask] = valid_output
        
        return output.view(*original_shape[:-1], -1)

class ImprovedGraphInteractionModel(nn.Module):
    def __init__(self, num_classes, locations, d_model = 256, nhead = 8, num_layers = 6, 
                 T_past = 10, T_future = 10, max_agents = 50, pad_value=-999.0):
        super().__init__()
        self.num_classes = num_classes
        self.locations = locations
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.T_past = T_past
        self.T_future = T_future
        self.max_agents = 50
        self.pad_value = pad_value  # Store pad value

        self.location_to_idx = {loc:i for i,loc in enumerate(locations)}

        self.class_embedding = nn.Embedding(num_classes, d_model)
        self.location_embedding = nn.Embedding(len(locations), d_model)

        # FIXED: Use safe spatial encoding
        self.spatial_encoding = SafeSpatialEncoding(d_model, pad_value=pad_value)
        self.temporal_encoding = PositionalEncoding(d_model)

        # FIXED: Use safe position projection
        self.pos_projection = SafePositionProjection(d_model, pad_value=pad_value)

        self.unobserved_token = nn.Parameter(torch.randn(self.d_model) * 0.01)
        self.gap_encoding = nn.Embedding(T_past + 1, d_model)

        self.transformer_encoder = AlternatingTransformerEncoder(
            self.d_model, self.nhead, self.num_layers, 4 * self.d_model
        )

        self.transformer_decoder = StandardTransformerDecoder(
            self.d_model, self.nhead, self.num_layers, 4 * self.d_model
        )

        self.output_norm = nn.LayerNorm(d_model)

        self.delta_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        self.uncertainty_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2)
        )

        self.position_interpolator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        self._initialize_weights()
        
        # IMPROVED: Learned parameters for stability (Trajectron++ approach)
        self.register_parameter('log_sigma_init', nn.Parameter(torch.tensor(-2.0)))
        self.register_parameter('max_delta_scale', nn.Parameter(torch.tensor(20.0)))

    def _initialize_weights(self):
        for module in self.modules():
            #xavier/glorot init
            if isinstance(module, nn.Linear): 
                nn.init.xavier_uniform_(module.weight, gain = 0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std = 0.1)

        nn.init.xavier_uniform_(self.delta_projection[-1].weight, gain = 0.01)
        nn.init.zeros_(self.delta_projection[-1].bias)

        nn.init.xavier_uniform_(self.uncertainty_projection[-1].weight, gain = 0.01)
        nn.init.constant_(self.uncertainty_projection[-1].bias, -2)

    def _create_enhanced_past_embeddings_safe(self, past_positions, past_positions_orig, obs_mask,
                                  location_emb, class_emb, batch, device):
        """FIXED: Safe enhanced embeddings creation"""
        batch_size, num_agents, seq_len, _ = past_positions.shape
        print(f"\n🎨 Creating SAFE enhanced embeddings...")
        print(f"   Input shapes: past_pos={past_positions.shape}, obs_mask={obs_mask.shape}")
        print(f"   Position range: [{past_positions.min().item():.6f}, {past_positions.max().item():.6f}]")
        print(f"   Pad value: {self.pad_value}")
        
        # Count how many positions are padded
        padded_positions = (past_positions == self.pad_value).any(dim=-1).sum().item()
        total_positions = batch_size * num_agents * seq_len
        print(f"   Padded positions: {padded_positions}/{total_positions} ({100*padded_positions/total_positions:.1f}%)")
        
        # Calculate gaps (same as before)
        gaps = torch.zeros_like(obs_mask, dtype=torch.long, device=device)
        for b in range(batch_size):
            for n in range(num_agents):
                if batch['agent_masks'][b, n] == 0:
                    continue
                    
                gap_counter = 0
                for t in range(seq_len):
                    if obs_mask[b, n, t] == 1:
                        gap_counter = 0
                    else:
                        gap_counter += 1
                    gaps[b, n, t] = min(gap_counter, self.T_past)
        
        # FIXED: Safe position projection that handles pad values
        print(f"   Applying SAFE position projection...")
        pos_emb = self.pos_projection(past_positions)
        
        if torch.isnan(pos_emb).any():
            print("🚨 NaN after SAFE position projection!")
            pos_emb = torch.nan_to_num(pos_emb, nan=0.0)
        else:
            print("✅ Position projection successful")
        
        # FIXED: Safe spatial encoding
        print(f"   Applying SAFE spatial encoding...")
        spatial_emb = self.spatial_encoding(past_positions)
        
        if torch.isnan(spatial_emb).any():
            print("🚨 NaN in SAFE spatial encoding!")
            spatial_emb = torch.nan_to_num(spatial_emb, nan=0.0)
        else:
            print("✅ Spatial encoding successful")
        
        # Only add spatial encoding where positions are valid (not padded)
        valid_positions_mask = ~((past_positions == self.pad_value).any(dim=-1, keepdim=True))
        pos_emb = pos_emb + spatial_emb * valid_positions_mask.float()
        
        # Gap encoding
        print(f"   Applying gap encoding...")
        gap_emb = self.gap_encoding(gaps)
        if torch.isnan(gap_emb).any():
            print("🚨 NaN in gap encoding!")
            gap_emb = torch.nan_to_num(gap_emb, nan=0.0)
        
        pos_emb = pos_emb + gap_emb
        
        # Context embeddings - only for existing agents
        print(f"   Adding context embeddings...")
        class_expanded = class_emb.unsqueeze(2).expand(-1, -1, seq_len, -1)
        location_expanded = location_emb.unsqueeze(1).unsqueeze(2).expand(-1, num_agents, seq_len, -1)
        agent_mask_expanded = batch['agent_masks'].unsqueeze(2).unsqueeze(3).expand(-1, -1, seq_len, self.d_model)
        
        # Only add context for existing agents
        pos_emb = pos_emb + (class_expanded + location_expanded) * agent_mask_expanded
        
        # Final agent masking - zero out non-existent agents completely
        pos_emb = pos_emb * agent_mask_expanded
        
        # Reshape and apply temporal encoding
        print(f"   Reshaping and applying temporal encoding...")
        pos_emb = pos_emb.view(batch_size * num_agents, seq_len, self.d_model)
        pos_emb = self.temporal_encoding(pos_emb)
        
        if torch.isnan(pos_emb).any():
            print("🚨 NaN after temporal encoding!")
            pos_emb = torch.nan_to_num(pos_emb, nan=0.0)
        
        print(f"   Final embedding stats:")
        print(f"   Shape: {pos_emb.shape}")
        print(f"   Range: [{pos_emb.min().item():.6f}, {pos_emb.max().item():.6f}]")
        print(f"   Mean: {pos_emb.mean().item():.6f}")
        print(f"   Std: {pos_emb.std().item():.6f}")
        print(f"   Contains NaN: {torch.isnan(pos_emb).any()}")
        print(f"   Contains Inf: {torch.isinf(pos_emb).any()}")
        
        return pos_emb

    def _teacher_forcing_predictions(self, memory, attention_mask, current_pos_est,
                               future_positions_target, location_emb, class_emb,
                               batch_size, num_agents, device, agent_masks=None):
        """IMPROVED: Following Trajectron++ approach with soft constraints"""
        current_pos_flat = current_pos_est.view(batch_size * num_agents, 2)
        future_pos_flat = future_positions_target.view(batch_size * num_agents, self.T_future, 2)

        decoder_positions = torch.cat([
            current_pos_flat.unsqueeze(1),
            future_pos_flat[:, :-1, :]
        ], dim=1)
        
        # FIXED: Use safe position projection for decoder input too
        decoder_input = self.pos_projection(decoder_positions)

        location_flat = location_emb.unsqueeze(1).expand(batch_size, num_agents, -1).contiguous()
        location_flat = location_flat.view(batch_size * num_agents, self.d_model)
        class_flat = class_emb.view(batch_size * num_agents, self.d_model)

        context_emb = class_flat + location_flat
        decoder_input = decoder_input + context_emb.unsqueeze(1)

        if agent_masks is not None:
            agent_mask_flat = agent_masks.view(batch_size * num_agents, 1, 1)
            decoder_input = decoder_input * agent_mask_flat

        future_temporal_emb = torch.zeros(self.T_future, self.d_model, device=device)
        max_pe_len = self.temporal_encoding.pe.size(0)

        for i in range(self.T_future):
            temporal_pos = self.T_past + i
            if temporal_pos < max_pe_len:
                future_temporal_emb[i] = self.temporal_encoding.pe[temporal_pos]

        decoder_input = decoder_input + future_temporal_emb.unsqueeze(0).expand(batch_size * num_agents, -1, -1)

        causal_mask = torch.triu(
            torch.ones(self.T_future, self.T_future, device=device) * float('-inf'), 
            diagonal=1
        )

        decoder_output = self.transformer_decoder(tgt = decoder_input, memory = memory, tgt_mask = causal_mask,
                                                  memory_key_padding_mask=attention_mask)
        # IMPROVED: Layer norm before output projections
        normalized_output = self.output_norm(decoder_output)

        # IMPROVED: Soft-constrained outputs using tanh scaling (Social-GAN style)
        delta_raw = self.delta_projection(normalized_output)
        delta_mu = torch.tanh(delta_raw) * self.max_delta_scale

        # IMPROVED: Conservative uncertainty prediction
        log_var_raw = self.uncertainty_projection(normalized_output)
        log_var = self.log_sigma_init + torch.tanh(log_var_raw) * 2.0  # Range: [log_sigma_init-2, log_sigma_init+2]

        # Convert deltas to positions
        positions_mu = self._convert_deltas_to_positions_teacher_forcing(
            current_pos_flat, delta_mu
        )

        # IMPROVED: Proper variance propagation without explosion
        delta_var = torch.exp(log_var)
        positions_var = self._propagate_variance_proper(delta_var)

        return {
            'future_deltas_mu': delta_mu.view(batch_size, num_agents, self.T_future, 2),
            'future_deltas_logvar': log_var.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_mu': positions_mu.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_var': positions_var.view(batch_size, num_agents, self.T_future, 2)
        }
    
    def _propagate_variance_proper(self, delta_var):
        """Proper variance propagation following probability theory"""
        # For independent Gaussians: Var(X + Y) = Var(X) + Var(Y)
        pos_var = torch.cumsum(delta_var, dim=1)  # Much cleaner!
        return pos_var
    
    def _generate_autoregressive_predictions(self, memory, attention_mask, current_pos_est,
                                           class_emb, location_emb, 
                                           batch_size, num_agents, device):
        """IMPROVED: Autoregressive generation with proper constraints"""
        
        current_pos = current_pos_est.view(batch_size * num_agents, 2)
        
        # Context embeddings
        class_flat = class_emb.view(batch_size * num_agents, self.d_model)
        location_flat = location_emb.unsqueeze(1).expand(batch_size, num_agents, -1).contiguous()
        location_flat = location_flat.view(batch_size * num_agents, self.d_model)
        context_emb = class_flat + location_flat
        
        # Storage for predictions
        future_deltas_mu = []
        future_deltas_logvar = []
        future_positions = []
        
        for t in range(self.T_future):
            # FIXED: Use safe position projection for autoregressive input
            decoder_input = self.pos_projection(current_pos.unsqueeze(1))
            decoder_input = decoder_input + context_emb.unsqueeze(1)
            
            # Add temporal encoding
            temporal_pos = self.T_past + t
            if temporal_pos < self.temporal_encoding.pe.size(0):
                temporal_emb = self.temporal_encoding.pe[temporal_pos]
                decoder_input = decoder_input + temporal_emb.unsqueeze(0).unsqueeze(0)
            
            # Apply decoder
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=None,
                memory_key_padding_mask=attention_mask
            )
            
            # IMPROVED: Same soft constraints as teacher forcing
            normalized_output = self.output_norm(decoder_output.squeeze(1))
            
            delta_raw = self.delta_projection(normalized_output)
            delta_mu = torch.tanh(delta_raw) * self.max_delta_scale
            
            log_var_raw = self.uncertainty_projection(normalized_output)
            log_var = self.log_sigma_init + torch.tanh(log_var_raw) * 2.0
            
            # Update position
            next_pos = current_pos + delta_mu
            
            # Store predictions
            future_deltas_mu.append(delta_mu)
            future_deltas_logvar.append(log_var)
            future_positions.append(next_pos)
            
            current_pos = next_pos
        
        # Stack predictions
        deltas_mu = torch.stack(future_deltas_mu, dim=1)
        deltas_logvar = torch.stack(future_deltas_logvar, dim=1)
        positions = torch.stack(future_positions, dim=1)
        
        # Proper variance propagation
        delta_var = torch.exp(deltas_logvar)
        positions_var = self._propagate_variance_proper(delta_var)
        
        return {
            'future_deltas_mu': deltas_mu.view(batch_size, num_agents, self.T_future, 2),
            'future_deltas_logvar': deltas_logvar.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_mu': positions.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_var': positions_var.view(batch_size, num_agents, self.T_future, 2)
        }
    
    def _convert_deltas_to_positions_teacher_forcing(self, start_pos, deltas):
        """Convert delta predictions to positions during teacher forcing"""
        positions = torch.zeros_like(deltas)  # (B*N, T_future, 2)
        current_pos = start_pos  # (B*N, 2)
        
        for i in range(deltas.size(1)):
            current_pos = current_pos + deltas[:, i, :]
            positions[:, i, :] = current_pos
        
        return positions
    
    def _estimate_current_position(self, memory, obs_mask, past_positions, agent_masks=None):
        """Estimate current position for each agent - FIXED to handle pad values"""
        batch_size, num_agents, seq_len, _ = past_positions.shape
        
        # Reshape memory: (B*N, T, d_model) -> (B, N, T, d_model)
        memory_reshaped = memory.view(batch_size, num_agents, seq_len, self.d_model)
        
        # Create attention weights, but only for valid (non-padded) positions
        valid_positions_mask = ~((past_positions == self.pad_value).any(dim=-1))  # (B, N, T)
        
        # Combine observation mask with valid positions mask
        combined_mask = obs_mask * valid_positions_mask.float()
        
        # Create attention weights
        weights = torch.softmax(memory_reshaped.mean(dim=-1), dim=-1)  # (B, N, T)
        weights = weights * combined_mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted combination
        context = torch.sum(memory_reshaped * weights.unsqueeze(-1), dim=2)  # (B, N, d_model)
        current_pos = self.position_interpolator(context)  # (B, N, 2)
        
        # Zero out positions for non-existent agents if agent_masks provided
        if agent_masks is not None:
            current_pos = current_pos * agent_masks.unsqueeze(-1)
        
        return current_pos

    def forward(self, batch, use_teacher_forcing=True):
        """Main forward pass with proper non-existent agent handling"""
        batch_size, num_agents = batch['past_positions'].shape[:2]
        device = batch['past_positions'].device
        
        # FIXED: Enhanced NaN checking with pad value awareness
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                print(f"WARNING: NaN detected in input {key}")
                batch[key] = torch.nan_to_num(tensor, nan=0.0)
        
        location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], 
                                    device=device)
        
        # Get embeddings
        class_emb = self._get_safe_embeddings(batch['agent_labels'], batch['agent_masks'])
        location_emb = self.location_embedding(location_indices)
        
        # FIXED: Use safe embedding creation
        past_emb = self._create_enhanced_past_embeddings_safe(
            batch['past_positions'], batch['past_positions_orig'], batch['obs_masks'], 
            location_emb, class_emb, batch, device
        )
        
        # Check for NaN after embeddings
        if torch.isnan(past_emb).any():
            print("WARNING: NaN in past embeddings after SAFE creation!")
            return None
        
        # Encode with proper masking
        memory = self.transformer_encoder(
            past_emb, 
            batch_size=batch_size, 
            num_agents=num_agents, 
            seq_len=self.T_past,
            obs_masks=batch['obs_masks'],  # Pass original masks
            agent_masks=batch['agent_masks']
        )
        
        # Check for NaN after encoding
        if torch.isnan(memory).any():
            print("WARNING: NaN in encoder output!")
            return None
        
        # Rest of forward pass...
        current_pos_est = self._estimate_current_position(
            memory, batch['obs_masks'], batch['past_positions'], batch['agent_masks']
        )
        
        if use_teacher_forcing and self.training:
            predictions = self._teacher_forcing_predictions(
                memory, None, current_pos_est, batch['future_positions'],  # Set attention_mask to None for now
                location_emb, class_emb, batch_size, num_agents, device, batch['agent_masks']
            )
        else:
            predictions = self._generate_autoregressive_predictions(
                memory, None, current_pos_est, class_emb, 
                location_emb, batch_size, num_agents, device
            )
        
        predictions['current_position_estimate'] = current_pos_est
        return predictions

    def _get_safe_embeddings(self, agent_labels, agent_masks):
        """Get embeddings only for existing agents"""
        existing_mask = (agent_labels != -1)
        safe_labels = torch.where(existing_mask, agent_labels, 0)
        class_emb_all = self.class_embedding(safe_labels)

        if torch.isnan(class_emb_all).any():
            print("WARNING: NaN in class embeddings!")
            class_emb_all = torch.nan_to_num(class_emb_all, nan=0.0)
        
        # Zero out embeddings for non-existent agents using agent_masks
        class_emb = class_emb_all * agent_masks.unsqueeze(-1).float()
        return class_emb
    
    def denormalize_positions(self, normalized_coords, video_stats):
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





    
