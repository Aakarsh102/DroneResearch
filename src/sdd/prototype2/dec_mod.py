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
        # x: (B, T, d_model)
        return x + self.pe[:x.size(1)].unsqueeze(0)

class SafeSpatialEncoding(nn.Module):
    """Spatial encoding that properly handles padding values"""
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
        
        # Learnable "pad embedding" for padded positions
        self.pad_embedding = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: (..., 2) — normalized x,y coordinates or pad_value for padding
        returns: (..., d_model)
        """
        original_shape = positions.shape
        positions_flat = positions.view(-1, 2)
        batch_flat_size = positions_flat.size(0)
        
        # Create mask for valid (non-padded) positions
        valid_mask = ~((positions_flat == self.pad_value).any(dim=1))
        
        # Initialize output with pad embeddings
        pe = self.pad_embedding.unsqueeze(0).expand(batch_flat_size, -1).clone()
        
        if valid_mask.any():
            # Only process valid positions
            valid_positions = positions_flat[valid_mask]
            
            # Clamp valid positions to reasonable range
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
            pe[valid_mask] = pe_valid
        
        return pe.view(*original_shape[:-1], self.d_model)

class SafePositionProjection(nn.Module):
    """Position projection that handles padding values"""
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

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal masking"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None, causal_mask=False):
        """
        x: (B, T, d_model)
        attention_mask: (B, T) - True where positions should be masked
        causal_mask: bool - whether to apply causal masking
        """
        B, T, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, T, head_dim)
        K = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, T, head_dim)
        V = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, T, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nhead, T, T)
        
        # Apply masks
        if causal_mask:
            causal_mask_tensor = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask_tensor.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T) for broadcasting
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            scores.masked_fill_(mask_expanded, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, nhead, T, head_dim)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)  # (B, T, d_model)
        out = self.out_proj(out)
        
        return out

class DecoderOnlyTrajectoryModel(nn.Module):
    """Decoder-only transformer for trajectory prediction with delta outputs"""
    
    def __init__(self, num_classes, locations, d_model=256, nhead=8, num_layers=6, 
                 T_past=10, T_future=10, max_agents=50, pad_value=-999.0):
        super().__init__()
        
        # Model configuration
        self.num_classes = num_classes
        self.locations = locations
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.T_past = T_past
        self.T_future = T_future
        self.max_agents = max_agents
        self.pad_value = pad_value
        self.total_seq_len = T_past + T_future
        self.causal_mask = self.build_causal_mask(self.T_past + self.T_future)
        
        # Location mapping
        self.location_to_idx = {loc: i for i, loc in enumerate(locations)}
        
        # Embeddings
        self.class_embedding = nn.Embedding(num_classes + 1, d_model)
        self.location_embedding = nn.Embedding(len(locations), d_model)
        
        # Position encodings
        self.position_projection = SafePositionProjection(d_model, pad_value=pad_value)
        self.spatial_encoding = SafeSpatialEncoding(d_model, pad_value=pad_value)
        self.temporal_encoding = PositionalEncoding(d_model, max_len=self.total_seq_len)
        self.gap_embedding = nn.Embedding(self.T_past, d_model) # how far in the future should the first predcition be
        
        # Special tokens
        self.unobserved_token = nn.Parameter(torch.randn(d_model) * 0.01)
        self.future_token = nn.Parameter(torch.randn(d_model) * 0.01)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=4*self.d_model,
                                                            dropout=0.1,batch_first = True)
        self.transformer_decoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        
        # Transformer decoder layers
        # self.layers = nn.ModuleList([
        #     TransformerDecoderLayer(d_model, nhead, 4 * d_model, dropout=0.1)
        #     for _ in range(num_layers)
        # ])
        
        # Output projections
        self.output_norm = nn.LayerNorm(d_model)
        
        # Position prediction head
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        
        # Uncertainty prediction head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2)
        )
        
        # Learned parameters for stability
        self.register_parameter('log_sigma_init', nn.Parameter(torch.tensor(-2.0)))
        self.register_parameter('max_position_scale', nn.Parameter(torch.tensor(20.0)))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)
                
        # Initialize output heads more conservatively
        nn.init.xavier_uniform_(self.position_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.position_head[-1].bias)
        
        nn.init.xavier_uniform_(self.uncertainty_head[-1].weight, gain=0.01)
        nn.init.constant_(self.uncertainty_head[-1].bias, -2)
    
    # def _get_last_valid_position(self, past_positions, obs_masks, agent_masks):
    #     """Get the last valid position for each agent to start delta computation"""
    #     batch_size, num_agents, T_past, _ = past_positions.shape
    #     device = past_positions.device
        
    #     # Initialize with zeros
    #     last_positions = torch.zeros(batch_size, num_agents, 2, device=device)
        
    #     for b in range(batch_size):
    #         for n in range(num_agents):
    #             if agent_masks[b, n] == 0:  # Non-existent agent
    #                 continue
                    
    #             # Find last observed position for this agent
    #             valid_times = obs_masks[b, n].nonzero(as_tuple=True)[0]
    #             if len(valid_times) > 0:
    #                 last_valid_time = valid_times[-1]
    #                 last_positions[b, n] = past_positions[b, n, last_valid_time]
    #             else:
    #                 print("wrong dimensions")
        
    #     return last_positions
    def _get_last_valid_position(self, past_positions, obs_masks, agent_masks):
        """
        Get the last valid position for each agent
        past_positions: (B, N, T, 2)
        obs_masks:      (B, N, T) - can be float or boolean (1=observed, 0=not observed)
        agent_masks:    (B, N) - can be float or boolean (1=exists, 0=doesn't exist)
        returns last_positions: (B, N, 2)
        """
        B, N, T, _ = past_positions.shape
        device = past_positions.device
        
        # Convert to boolean if necessary
        obs_masks_bool = obs_masks.bool()  # Convert float to boolean
        agent_masks_bool = agent_masks.bool()  # Convert float to boolean
        
        # 1) Build time index grid [0,1,2,...,T-1] → (B,N,T)
        time_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, N, T)
        
        # 2) Mask out unobserved positions (set to -1)
        masked_idx = torch.where(obs_masks_bool, time_idx, torch.full_like(time_idx, -1))
        
        # 3) Find last observed time index (-1 if none observed)
        last_idx = masked_idx.amax(dim=2)  # (B, N), values in [-1 .. T-1]
        
        # 4) For agents with no observations, use index 0 (but will be zeroed out later)
        last_idx_safe = last_idx.clamp(min=0)  # (B, N)
        
        # 5) Create batch and agent indices for advanced indexing
        b_idx = torch.arange(B, device=device)[:, None].expand(B, N)  # (B, N)
        n_idx = torch.arange(N, device=device)[None, :].expand(B, N)  # (B, N)
        
        # 6) Gather positions at last valid indices
        last_positions = past_positions[b_idx, n_idx, last_idx_safe]  # (B, N, 2)
        
        # 7) Create mask for agents that actually had observations
        has_obs = (last_idx >= 0)  # (B, N) boolean
        valid_agent_and_obs = agent_masks_bool & has_obs  # (B, N) boolean
        
        # 8) Zero out positions for non-existent agents or agents with no observations
        last_positions = last_positions * valid_agent_and_obs.unsqueeze(-1).float()
        
        return last_positions
    
    def _compute_target_deltas(self, past_positions, future_positions, obs_masks, agent_masks):
        """Compute target deltas for training"""
        batch_size, num_agents, T_past, _ = past_positions.shape
        T_future = future_positions.shape[2]
        device = past_positions.device
        
        # Get last valid positions from past
        last_positions = self._get_last_valid_position(past_positions, obs_masks, agent_masks)
        
        # Compute deltas for future positions
        target_deltas = torch.zeros(batch_size, num_agents, T_future, 2, device=device)
        
        # First future step: delta from last past position
        target_deltas[:, :, 0] = future_positions[:, :, 0] - last_positions
        
        # Subsequent future steps: delta from previous future position
        for t in range(1, T_future):
            target_deltas[:, :, t] = future_positions[:, :, t] - future_positions[:, :, t-1]
        
        return target_deltas
    
    def _deltas_to_positions(self, deltas, past_positions, obs_masks, agent_masks):
        """Convert predicted deltas to absolute positions using cumulative sum"""
        batch_size, num_agents, T_future, _ = deltas.shape
        device = deltas.device
        
        # Get starting positions (last valid position from past)
        start_positions = self._get_last_valid_position(past_positions, obs_masks, agent_masks)
        
        # Initialize future positions
        future_positions = torch.zeros_like(deltas)
        
        # First future position: start_position + first_delta
        future_positions[:, :, 0] = start_positions + deltas[:, :, 0]
        
        # Subsequent positions: cumulative sum of deltas
        for t in range(1, T_future):
            future_positions[:, :, t] = future_positions[:, :, t-1] + deltas[:, :, t]
        
        # Apply agent masking
        agent_mask_expanded = agent_masks.unsqueeze(2).unsqueeze(3).expand(-1, -1, T_future, 2)
        future_positions = future_positions * agent_mask_expanded
        
        return future_positions
    
    def build_causal_mask(self, T: int, device: torch.device = None) -> torch.Tensor:
        """
        Returns a boolean mask of shape (T, T) where mask[i, j] = True 
        if j > i, i.e. token i cannot attend to token j (the “future”).
        """
        if device is None:
            # infer device from default tensor type
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ones above the diagonal → True entries where we want to block attention
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask
    
    def compute_gaps(self,obs_masks):
        """
        Compute gap from last observed position to end of sequence
        obs_masks: (B, N, T) - can be float or boolean (1 = observed, 0 = not observed)
        returns:   gaps: (B, N) integer in [0..T]
        """
        B, N, T = obs_masks.shape
        device = obs_masks.device
        
        # Convert to boolean if necessary
        obs_masks_bool = obs_masks.bool()
        
        # Create time indices [0, 1, 2, ..., T-1]
        time_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, N, T)
        
        # Mask out unobserved positions (set to -1)
        masked_idx = torch.where(obs_masks_bool, time_idx, torch.full_like(time_idx, -1))
        
        # Find last observed time index for each agent
        last_obs_idx = masked_idx.amax(dim=2)  # (B, N), values in [-1 .. T-1]
        
        # Detect sequences with no observations
        any_obs = obs_masks_bool.any(dim=2)  # (B, N) boolean
        
        # Compute gap: T - (last_obs_idx + 1) = T - last_obs_idx - 1
        # But if no observations, gap = T
        gaps = torch.where(
            any_obs,
            T - last_obs_idx - 1,  # Gap from last observation to end
            torch.full_like(last_obs_idx, T)  # If no obs, entire sequence is gap
        )
        
        # Clamp to valid range [0, T-1] for embedding indices
        gaps = gaps.clamp(0, T-1)
        
        return gaps
    
    def _create_sequence_embeddings(self, batch, device):
        """Create embeddings for the full sequence (past + future)"""
        batch_size, num_agents = batch['past_positions'].shape[:2]
        
        # Get context embeddings
        location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], device=device)
        class_emb = self._get_safe_class_embeddings(batch['agent_labels'], batch['agent_masks'])
        location_emb = self.location_embedding(location_indices)
        
        # Create full sequence positions (past + future placeholders)
        past_positions = batch['past_positions']  # (B, N, T_past, 2)
        
        if self.training:
            # During training, use ground truth future positions
            future_positions = batch['future_positions']  # (B, N, T_future, 2)
        else:
            # During inference, use pad values for future positions (will be filled iteratively)
            future_positions = torch.full (
                (batch_size, num_agents, self.T_future, 2), 
                self.pad_value, device=device
            )
        
        # Concatenate past and future
        full_positions = torch.cat([past_positions, future_positions], dim=2)  # (B, N, T_past+T_future, 2)
        
        # Create position embeddings
        pos_emb = self.position_projection(full_positions)  # (B, N, T_total, d_model)
        
        # Add spatial encoding (only for valid positions)
        # spatial_emb = self.spatial_encoding(full_positions)
        valid_mask = ~((full_positions == self.pad_value).any(dim=-1, keepdim=True))
        # pos_emb = pos_emb + spatial_emb * valid_mask.float()
        
        # Add context embeddings
        class_emb_expanded = class_emb.unsqueeze(2).expand(-1, -1, self.total_seq_len, -1)
        location_emb_expanded = location_emb.unsqueeze(1).unsqueeze(2).expand(-1, num_agents, self.total_seq_len, -1)
        
        pos_emb = pos_emb + class_emb_expanded + location_emb_expanded
        
        # Handle unobserved positions in the past
        obs_mask = batch['obs_masks']  # (B, N, T_past)
        
        # Create full observation mask (past observations + future as unobserved during training)
        if self.training:
            future_obs_mask = torch.zeros(batch_size, num_agents, self.T_future, device=device)
        else:
            future_obs_mask = torch.zeros(batch_size, num_agents, self.T_future, device=device)
        
        full_obs_mask = torch.cat([obs_mask, future_obs_mask], dim=2)  # (B, N, T_total)
        
        # Replace unobserved positions with learned token
        unobserved_positions = (full_obs_mask == 0)
        unobserved_past = (obs_mask == 0)
        pos_emb[unobserved_past] = self.unobserved_token
        
        
        # Apply agent masking
        # gaps = torch.zeros(batch_size, num_agents, dtype=torch.long, device=device)
        # for b in range(len(obs_mask)):
        #     for n in range(len(b)):
        #         if batch['agent_masks'][b, n] == 0:
        #             continue
        #         gap_counter = 0
        #         for t in range(self.T_past):
        #             if obs_mask[b, n, t] == 1:
        #                 gap_counter = 0
        #             else:
        #                 gap_counter += 1
        #         gaps[b, n] = min(gap_counter, self.T_past)

        gaps = self.compute_gaps(obs_mask)  # obs_mask is (B, N, T_past)
        gap_emb = self.gap_embedding(gaps).unsqueeze(2).expand(-1, -1, self.T_past, -1)
        pos_emb[:, :, :self.T_past] += gap_emb  # Add only to past positions
        # enc = self.gap_embedding(gaps).unsqueeze(2).expand(-1, -1, self.T_past,-1)
        # pos_emb += enc

        agent_mask_expanded = batch['agent_masks'].unsqueeze(2).unsqueeze(3).expand(-1, -1, self.total_seq_len, self.d_model)
        pos_emb = pos_emb * agent_mask_expanded
        
        return pos_emb, full_positions, full_obs_mask
    
    def _get_safe_class_embeddings(self, agent_labels, agent_masks):
        """Get class embeddings safely handling non-existent agents"""
        # existing_mask = (agent_labels != -1)
        # safe_labels = torch.where(existing_mask, agent_labels, 0)
        safe_labels = agent_labels.clone().long() + 1      # -1→0, 0→1, …, C-1→C
        class_emb   = self.class_embedding(safe_labels)    # padding_idx=0 is safe
        return class_emb * agent_masks.unsqueeze(-1) 
        # class_emb = self.class_embedding(agent_labels)
        
        # Zero out embeddings for non-existent agents
        # class_emb = class_emb * agent_masks.unsqueeze(-1).float()
        # return class_emb
    
    def _create_attention_mask(self, batch, full_obs_mask=None, is_causal=True):
        """Create attention mask for the sequence"""
        batch_size, num_agents = batch['past_positions'].shape[:2]
        seq_len = self.total_seq_len  # T_past + T_future
        
        # Initialize attention mask (True where positions should be masked)
        attention_mask = torch.zeros(batch_size, num_agents, seq_len, dtype=torch.bool, 
                                    device=batch['past_positions'].device)
        
        # 1. Agent mask: if agent_masks = 0, mask entire sequence for that agent
        agent_mask = batch['agent_masks']  # (B, N)
        non_existent_agents = (agent_mask == 0)
        attention_mask = attention_mask | non_existent_agents.unsqueeze(2).expand(-1, -1, seq_len)
        
        # 2. Past timesteps: use obs_masks for past positions
        obs_mask = batch['obs_masks']  # (B, N, T_past) - only for past
        unobserved_past = (obs_mask == 0)
        attention_mask[:, :, :self.T_past] = attention_mask[:, :, :self.T_past] | unobserved_past
        
        # 3. Temporal masks for past and future
        temporal_mask_past = batch['temporal_masks_past']  # (B, N, T_past)
        temporal_mask_future = batch['temporal_masks_future']  # (B, N, T_future)
        
        # Mask timesteps where temporal_mask = 1 (agent does exist at this timestep)
        invalid_past = (temporal_mask_past != 1)
        invalid_future = (temporal_mask_future != 1)
        
        attention_mask[:, :, :self.T_past] = attention_mask[:, :, :self.T_past] | invalid_past
        attention_mask[:, :, self.T_past:] = attention_mask[:, :, self.T_past:] | invalid_future
        
        # 4. Handle completely masked agents to prevent NaN in attention
        # Check if any agent (N dimension) is completely masked
        completely_masked = attention_mask.all(dim=2)  # (B, N) - True if agent is completely masked
        
        # For completely masked agents, unmask the first timestep to prevent NaN
        first_timestep_unmask = completely_masked.unsqueeze(2)  # (B, N, 1)
        attention_mask[:, :, 0:1] = attention_mask[:, :, 0:1] & ~first_timestep_unmask
        
        return attention_mask
    
    def forward(self, batch, use_teacher_forcing=True):
        """Forward pass of the decoder-only model"""
        device = batch['past_positions'].device
        batch_size, num_agents = batch['past_positions'].shape[:2]
        
        # Handle NaN inputs
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                print(f"WARNING: NaN detected in input {key}")
                batch[key] = torch.nan_to_num(tensor, nan=0.0)
        
        if self.training and use_teacher_forcing:
            return self._forward_teacher_forcing(batch, device)
        else:
            return self._forward_autoregressive(batch, device)
    
    def _forward_teacher_forcing(self, batch, device):
        """Forward pass with teacher forcing (training) - predicts deltas"""
        batch_size, num_agents = batch['past_positions'].shape[:2]
        device = batch['past_positions'].device
        
        # Create sequence embeddings
        seq_emb, full_positions, full_obs_mask = self._create_sequence_embeddings(batch, device)
        
        # Reshape for processing: (B, N, T, d_model) -> (B*N, T, d_model)
        seq_emb_flat = seq_emb.view(batch_size * num_agents, self.total_seq_len, self.d_model)
        
        # Add temporal encoding
        seq_emb_flat = self.temporal_encoding(seq_emb_flat)
        
        # Create attention mask
        attention_mask = self._create_attention_mask(batch, full_obs_mask)
        attention_mask_flat = attention_mask.view(batch_size * num_agents, self.total_seq_len)
        
        # Pass through transformer layers
        hidden_states = self.transformer_decoder(seq_emb_flat, mask = self.causal_mask, src_key_padding_mask=attention_mask_flat, is_causal=True)
        # for layer in self.layers:
        #     hidden_states = layer(
        #         hidden_states, 
        #         attention_mask=attention_mask_flat,
        #         causal_mask=True  # Use causal masking for autoregressive generation
        #     )
        
        # Normalize output
        hidden_states = self.output_norm(hidden_states)
        
        # Extract future timesteps for prediction
        future_hidden = hidden_states[:, self.T_past:, :]  # (B*N, T_future, d_model)
        
        # Predict positions and uncertainties
        position_raw = self.position_head(future_hidden)
        position_deltas = torch.tanh(position_raw) * 2.0  # Predict deltas with smaller scale
        
        uncertainty_raw = self.uncertainty_head(future_hidden)
        uncertainty_pred = self.log_sigma_init + torch.tanh(uncertainty_raw) * 2.0
        
        # Reshape outputs
        position_deltas = position_deltas.view(batch_size, num_agents, self.T_future, 2)
        uncertainty_pred = uncertainty_pred.view(batch_size, num_agents, self.T_future, 2)
        
        # Convert deltas to absolute positions
        position_pred = self._deltas_to_positions(
            position_deltas, 
            batch['past_positions'], 
            batch['obs_masks'], 
            batch['agent_masks']
        )
        
        # Apply agent masking
        agent_mask_expanded = batch['agent_masks'].unsqueeze(2).unsqueeze(3).expand(-1, -1, self.T_future, 2)
        position_pred = position_pred * agent_mask_expanded
        uncertainty_pred = uncertainty_pred * agent_mask_expanded
        
        return {
            'future_positions_mu': position_pred,
            'future_positions_logvar': uncertainty_pred,
            'future_positions_var': torch.exp(uncertainty_pred)
        }
    


    def _forward_autoregressive(self, batch, device):
        """Forward pass with autoregressive generation (inference) - predicts deltas"""
        batch_size, num_agents = batch['past_positions'].shape[:2]
        
        # Initialize with past positions
        current_positions = batch['past_positions'].clone()  # (B, N, T_past, 2)
        
        # Get starting positions for delta computation
        start_positions = self._get_last_valid_position(
            current_positions, batch['obs_masks'], batch['agent_masks']
        )
        
        # Storage for predictions
        future_deltas = []
        future_positions = []
        future_uncertainties = []
        
        # Current position tracker for delta computation
        current_position = start_positions.clone()  # (B, N, 2)
        
        # Get context embeddings (computed once)
        location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], device=device)
        class_emb = self._get_safe_class_embeddings(batch['agent_labels'], batch['agent_masks'])
        location_emb = self.location_embedding(location_indices)
        
        for t in range(self.T_future):
            # Create current sequence (past + generated future so far)
            if t == 0:
                seq_positions = current_positions  # Just past positions
                current_seq_len = self.T_past
            else:
                # Concatenate past + generated future
                generated_future = torch.stack(future_positions, dim=2)  # (B, N, t, 2)
                seq_positions = torch.cat([current_positions, generated_future], dim=2)
                current_seq_len = self.T_past + t
            
            # Create embeddings for current sequence
            pos_emb = self.position_projection(seq_positions)
            
            # Add context embeddings
            class_emb_expanded = class_emb.unsqueeze(2).expand(-1, -1, current_seq_len, -1)
            location_emb_expanded = location_emb.unsqueeze(1).unsqueeze(2).expand(-1, num_agents, current_seq_len, -1)
            pos_emb = pos_emb + class_emb_expanded + location_emb_expanded
            
            # Handle unobserved positions in past (only for t=0, for future steps this is handled by masking)
            if t == 0:
                obs_mask = batch['obs_masks']  # (B, N, T_past)
                unobserved_positions = (obs_mask == 0)
                pos_emb[unobserved_positions] = self.unobserved_token
                
                # Handle temporal mask for past
                temporal_mask_past = batch['temporal_masks_past']  # (B, N, T_past)
                invalid_past = (temporal_mask_past != 1)
                pos_emb[:, :, :self.T_past][invalid_past] = self.unobserved_token
            
            # Apply agent masking
            agent_mask_expanded = batch['agent_masks'].unsqueeze(2).unsqueeze(3).expand(-1, -1, current_seq_len, self.d_model)
            pos_emb = pos_emb * agent_mask_expanded
            
            # Reshape and add temporal encoding
            pos_emb_flat = pos_emb.view(batch_size * num_agents, current_seq_len, self.d_model)
            pos_emb_flat = self.temporal_encoding(pos_emb_flat)
            
            # Create proper attention mask
            attention_mask = torch.zeros(batch_size, num_agents, current_seq_len, dtype=torch.bool, device=device)
            
            # Agent mask: if agent_masks = 0, mask entire sequence for that agent
            non_existent_agents = (batch['agent_masks'] == 0)
            attention_mask = attention_mask | non_existent_agents.unsqueeze(2).expand(-1, -1, current_seq_len)
            
            if t == 0:
                # Only past positions
                obs_mask = batch['obs_masks']  # (B, N, T_past)
                unobserved_past = (obs_mask == 0)
                attention_mask[:, :, :self.T_past] = attention_mask[:, :, :self.T_past] | unobserved_past
                
                # Temporal mask for past
                temporal_mask_past = batch['temporal_masks_past']  # (B, N, T_past)
                invalid_past = (temporal_mask_past != 1)
                attention_mask[:, :, :self.T_past] = attention_mask[:, :, :self.T_past] | invalid_past
            else:
                # Past + some future positions
                obs_mask = batch['obs_masks']  # (B, N, T_past)
                unobserved_past = (obs_mask == 0)
                attention_mask[:, :, :self.T_past] = attention_mask[:, :, :self.T_past] | unobserved_past
                
                # Temporal mask for past
                temporal_mask_past = batch['temporal_masks_past']  # (B, N, T_past)
                invalid_past = (temporal_mask_past != 1)
                attention_mask[:, :, :self.T_past] = attention_mask[:, :, :self.T_past] | invalid_past
                
                # Temporal mask for generated future positions
                temporal_mask_future = batch['temporal_masks_future']  # (B, N, T_future)
                invalid_future = (temporal_mask_future != 1)
                attention_mask[:, :, self.T_past:] = attention_mask[:, :, self.T_past:] | invalid_future[:, :, :t]
            
            # Handle completely masked agents to prevent NaN in attention
            completely_masked = attention_mask.all(dim=2)  # (B, N)
            first_timestep_unmask = completely_masked.unsqueeze(2)  # (B, N, 1)
            attention_mask[:, :, 0:1] = attention_mask[:, :, 0:1] & ~first_timestep_unmask
            
            attention_mask_flat = attention_mask.view(batch_size * num_agents, current_seq_len)
            
            # Pass through transformer
            hidden_states = self.transformer_decoder(pos_emb_flat,src_key_padding_mask=attention_mask_flat)
            # for layer in self.layers:
            #     hidden_states = layer(
            #         hidden_states,
            #         attention_mask=attention_mask_flat,
            #         causal_mask=True
            #     )
            
            # Get prediction for next timestep (last position in sequence)
            next_hidden = hidden_states[:, -1:, :]  # (B*N, 1, d_model)
            next_hidden = self.output_norm(next_hidden)
            
            # Predict next position (as delta)
            position_raw = self.position_head(next_hidden)
            next_delta = torch.tanh(position_raw) * 2.0  # Predict delta with smaller scale
            
            uncertainty_raw = self.uncertainty_head(next_hidden)
            next_uncertainty = self.log_sigma_init + torch.tanh(uncertainty_raw) * 2.0
            
            # Reshape
            next_delta = next_delta.view(batch_size, num_agents, 2)
            next_uncertainty = next_uncertainty.view(batch_size, num_agents, 2)
            
            # Convert delta to absolute position
            next_position = current_position + next_delta
            
            # Apply agent masking
            agent_mask_2d = batch['agent_masks'].unsqueeze(2).expand(-1, -1, 2)
            next_position = next_position * agent_mask_2d
            next_uncertainty = next_uncertainty * agent_mask_2d
            
            # Update current position for next iteration
            current_position = next_position.clone()
            
            future_positions.append(next_position)
            future_uncertainties.append(next_uncertainty)
        
        # Stack predictions
        future_positions_tensor = torch.stack(future_positions, dim=2)  # (B, N, T_future, 2)
        future_uncertainties_tensor = torch.stack(future_uncertainties, dim=2)  # (B, N, T_future, 2)
        
        return {
            'future_positions_mu': future_positions_tensor,
            'future_positions_logvar': future_uncertainties_tensor,
            'future_positions_var': torch.exp(future_uncertainties_tensor)
        }
