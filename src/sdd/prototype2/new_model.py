import torch
import torch.nn as nn
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

class SpatialEncoding(nn.Module):
    def __init__(self, d_model, max_pos=10000):
        super(SpatialEncoding, self).__init__()
        self.d_model = d_model
        self.max_pos = max_pos
        self.half_dim = d_model//2

        div_term = torch.exp(
            torch.arange(0, self.half_dim, 2).float() *
            (-math.log(10000.0) / self.half_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: (..., 2) — normalized x,y in [0,1]
        returns: (..., d_model)
        """
        original_shape = positions.shape
        positions_flat = positions.view(-1, 2)
        batch_flat_size = positions_flat.size(0)
        
        scaled = positions_flat * self.max_pos
        
        pe_x = torch.zeros(batch_flat_size, self.half_dim, device=positions.device)
        pe_x[:, 0::2] = torch.sin(scaled[:, 0:1] * self.div_term.unsqueeze(0))
        pe_x[:, 1::2] = torch.cos(scaled[:, 0:1] * self.div_term.unsqueeze(0))

        pe_y = torch.zeros(batch_flat_size, self.half_dim, device=positions.device)
        pe_y[:, 0::2] = torch.sin(scaled[:, 1:2] * self.div_term.unsqueeze(0))
        pe_y[:, 1::2] = torch.cos(scaled[:, 1:2] * self.div_term.unsqueeze(0))
        
        pe = torch.cat([pe_x, pe_y], dim=-1)
        return pe.view(*original_shape[:-1], self.d_model)

class AlternatingTransformerEncoder(nn.Module):
    """Encoder that alternates between temporal and agent attention"""
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
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            ),
            'agent': nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
        })
    
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
    
    def _reshape_mask_for_agent_attention(self, mask, batch_size, num_agents, seq_len):
        """Reshape mask from (B*N, T) to (B*T, N) for agent attention"""
        if mask is None:
            return None
        # mask: (B*N, T) -> (B, N, T) -> (B, T, N) -> (B*T, N)
        mask = mask.view(batch_size, num_agents, seq_len)
        mask = mask.permute(0, 2, 1).contiguous()
        mask = mask.view(batch_size * seq_len, num_agents)
        return mask
    
    def _reshape_mask_for_temporal_attention(self, mask, batch_size, num_agents, seq_len):
        """Reshape mask from (B*T, N) to (B*N, T) for temporal attention"""
        if mask is None:
            return None
        # mask: (B*T, N) -> (B, T, N) -> (B, N, T) -> (B*N, T)
        mask = mask.view(batch_size, seq_len, num_agents)
        mask = mask.permute(0, 2, 1).contiguous()
        mask = mask.view(batch_size * num_agents, seq_len)
        return mask
        
    def forward(self, x, batch_size, num_agents, seq_len, attention_mask=None):
        """
        x: (B*N, T, d_model) - input format
        attention_mask: (B*N, T) - padding mask
        Returns: (B*N, T, d_model)
        """
        current_x = x
        current_mask = attention_mask
        
        for i, layer_dict in enumerate(self.layers):
            if i % 2 == 0:
                # Even layer: temporal attention (B*N, T, d_model)
                layer = layer_dict['temporal']
                
                # Ensure correct format for temporal attention
                current_x = self._ensure_temporal_format(current_x, batch_size, num_agents, seq_len)
                
                # Make sure mask is in temporal format (B*N, T)
                if current_mask is not None and current_mask.shape != (batch_size * num_agents, seq_len):
                    current_mask = self._reshape_mask_for_temporal_attention(
                        current_mask, batch_size, num_agents, seq_len
                    )
                    
            else:
                # Odd layer: agent attention (B*T, N, d_model)
                layer = layer_dict['agent']
                
                # Ensure correct format for agent attention
                current_x = self._ensure_agent_format(current_x, batch_size, num_agents, seq_len)
                
                # Reshape mask for agent attention (B*T, N)
                if current_mask is not None:
                    current_mask = self._reshape_mask_for_agent_attention(
                        current_mask, batch_size, num_agents, seq_len
                    )
            
            # Apply the layer
            current_x = layer(current_x, src_key_padding_mask=current_mask)
        
        # Ensure final output is in (B*N, T, d_model) format
        current_x = self._ensure_temporal_format(current_x, batch_size, num_agents, seq_len)
        
        return current_x

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

class GraphInteractionModel(nn.Module):
    def __init__(self, num_classes, locations, d_model=256, nhead=8, num_layers=6,
                 T_past=10, T_future=10, max_agents=50):
        super(GraphInteractionModel, self).__init__()
        
        self.d_model = d_model
        self.T_past = T_past
        self.T_future = T_future
        self.num_classes = num_classes
        self.max_agents = max_agents
        self.locations = locations
        self.location_to_idx = {loc:i for i,loc in enumerate(locations)}

        # Embeddings (NO agent embedding)
        self.class_embedding = nn.Embedding(num_classes, d_model)
        self.location_embedding = nn.Embedding(len(locations), d_model)
        
        # Encodings
        self.spatial_encoding = SpatialEncoding(d_model)
        self.temporal_encoding = PositionalEncoding(d_model)
        
        # Projections
        self.pos_projection = nn.Linear(2, d_model)
        
        # Special tokens
        self.unobserved_token = nn.Parameter(torch.randn(d_model))
        self.gap_encoding = nn.Embedding(T_past + 1, d_model)
        
        # Transformers - Modified to alternating encoder and standard decoder
        self.transformer_encoder = AlternatingTransformerEncoder(
            d_model, nhead, num_layers, d_model * 4
        )
        
        self.transformer_decoder = StandardTransformerDecoder(
            d_model, nhead, num_layers, d_model * 4
        )
        
        # Modified output heads - delta_x and logvar (not delta_logvar)
        self.delta_head = nn.Linear(d_model, 2)  # delta_x, delta_y
        self.logvar_head = nn.Linear(d_model, 2)  # log_var_x, log_var_y
        
        self.position_interpolator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

    def _create_enhanced_past_embeddings(self, past_positions, past_positions_orig, obs_mask,
                                    location_emb, class_emb, batch, device):
        """Create enhanced embeddings with proper agent masking"""
        batch_size, num_agents, seq_len, _ = past_positions.shape
        
        # Calculate gaps correctly (same as before)
        gaps = torch.zeros_like(obs_mask, dtype=torch.long, device=device)
        for b in range(batch_size):
            for n in range(num_agents):
                # Skip non-existent agents
                if batch['agent_masks'][b, n] == 0:
                    continue
                    
                gap_counter = 0
                for t in range(seq_len):
                    if obs_mask[b, n, t] == 1:
                        gap_counter = 0
                    else:
                        gap_counter += 1
                    gaps[b, n, t] = min(gap_counter, self.T_past)
        
        # Rest of the method stays the same, but add agent masking at the end
        pos_emb = torch.zeros(batch_size, num_agents, seq_len, self.d_model, device=device)
        location_expanded = location_emb.unsqueeze(1).unsqueeze(2).expand(-1, num_agents, seq_len, -1)

        # observed_mask = obs_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
        observed_mask = (obs_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model) == 1.0)
        
        observed_pos_emb = self.pos_projection(past_positions)
        unobserved_emb = self.unobserved_token.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_agents, seq_len, -1)
        
        pos_emb = torch.where(observed_mask, observed_pos_emb, unobserved_emb)
        
        spatial_emb = self.spatial_encoding(past_positions_orig)
        pos_emb = pos_emb + spatial_emb * observed_mask
        
        gap_emb = self.gap_encoding(gaps)
        pos_emb = pos_emb + gap_emb
        
        # Add context embeddings with agent masking
        class_expanded = class_emb.unsqueeze(2).expand(-1, -1, seq_len, -1)
        agent_mask_expanded = batch['agent_masks'].unsqueeze(2).unsqueeze(3).expand(-1, -1, seq_len, self.d_model)
        
        # Apply agent mask to class embeddings
        class_expanded_masked = class_expanded * agent_mask_expanded
        pos_emb = pos_emb + class_expanded_masked + location_expanded
        
        # Apply agent mask to final embeddings
        pos_emb = pos_emb * agent_mask_expanded
        
        # Reshape for transformer
        pos_emb = pos_emb.view(batch_size * num_agents, seq_len, self.d_model)
        pos_emb = self.temporal_encoding(pos_emb)
        
        return pos_emb

    def _teacher_forcing_predictions(self, memory, attention_mask, current_pos_est,
                               future_positions_target, location_emb, class_emb,
                               batch_size, num_agents, device, agent_masks=None):
        """Teacher forcing for training with modified output heads"""
        current_pos_flat = current_pos_est.view(batch_size * num_agents, 2)
        future_pos_flat = future_positions_target.view(batch_size * num_agents, self.T_future, 2)
        
        # Create decoder input sequence
        decoder_positions = torch.cat([
            current_pos_flat.unsqueeze(1),
            future_pos_flat[:, :-1, :]
        ], dim=1)  # (B*N, T_future, 2)
        
        # Project to embedding space
        decoder_input = self.pos_projection(decoder_positions)  # (B*N, T_future, d_model)
        
        # Add context embeddings (NO agent embedding)
        location_flat = location_emb.unsqueeze(1).expand(batch_size, num_agents, -1).contiguous()
        location_flat = location_flat.view(batch_size * num_agents, self.d_model)
        class_flat = class_emb.view(batch_size * num_agents, self.d_model)
        
        context_emb = location_flat + class_flat  # (B*N, d_model)
        decoder_input = decoder_input + context_emb.unsqueeze(1)
        
        # Apply agent mask to decoder input if provided
        if agent_masks is not None:
            agent_mask_flat = agent_masks.view(batch_size * num_agents, 1, 1)
            decoder_input = decoder_input * agent_mask_flat
        
        # Create temporal encoding for future steps
        future_temporal_emb = torch.zeros(self.T_future, self.d_model, device=device)
        max_pe_len = self.temporal_encoding.pe.size(0)
        
        for i in range(self.T_future):
            temporal_pos = self.T_past + i
            if temporal_pos < max_pe_len:
                future_temporal_emb[i] = self.temporal_encoding.pe[temporal_pos]
        
        # Add temporal encoding
        decoder_input = decoder_input + future_temporal_emb.unsqueeze(0).expand(batch_size * num_agents, -1, -1)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(self.T_future, self.T_future, device=device) * float('-inf'), 
            diagonal=1
        )
        
        # Apply decoder (standard format throughout)
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=attention_mask
        )
        
        # Generate predictions with separate heads
        delta_mu = self.delta_head(decoder_output)  # (B*N, T_future, 2)
        log_var = self.logvar_head(decoder_output)  # (B*N, T_future, 2)
        
        # Convert deltas to positions
        positions_mu = self._convert_deltas_to_positions_teacher_forcing(
            current_pos_flat, delta_mu
        )
        # Convert log variance to variance for position propagation
        delta_var = torch.exp(log_var)
        positions_var = self._propagate_variance(delta_var)
        
        return {
            'future_deltas_mu': delta_mu.view(batch_size, num_agents, self.T_future, 2),
            'future_deltas_logvar': log_var.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_mu': positions_mu.view(batch_size, num_agents, self.T_future, 2),
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
    
    def _propagate_variance(self, delta_var):
        """Propagate variance through cumulative sum"""
        pos_var = torch.zeros_like(delta_var)
        pos_var[:, 0] = delta_var[:, 0]
        
        for i in range(1, delta_var.size(1)):
            pos_var[:, i] = pos_var[:, i-1] + delta_var[:, i]
        
        return pos_var
    
    def _estimate_current_position(self, memory, obs_mask, past_positions, agent_masks=None):
        """Estimate current position for each agent"""
        batch_size, num_agents, seq_len, _ = past_positions.shape
        
        # Reshape memory: (B*N, T, d_model) -> (B, N, T, d_model)
        memory_reshaped = memory.view(batch_size, num_agents, seq_len, self.d_model)
        
        # Create attention weights
        weights = torch.softmax(memory_reshaped.mean(dim=-1), dim=-1)  # (B, N, T)
        weights = weights * obs_mask
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
        location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], 
                                    device=device)
        
        # Get embeddings - SAFE VERSION for non-existent agents
        class_emb = self._get_safe_embeddings(batch['agent_labels'], batch['agent_masks'])
        location_emb = self.location_embedding(location_indices)
        
        # Create past embeddings with agent masking
        past_emb = self._create_enhanced_past_embeddings(
            batch['past_positions'], batch['past_positions_orig'], batch['obs_masks'], 
            location_emb, class_emb, batch, device
        )
        
        # Create attention mask - FIXED VERSION
        obs_mask_flat = batch['obs_masks'].view(batch_size * num_agents, self.T_past)
        
        # Expand agent masks properly
        agent_mask_expanded = batch['agent_masks'].unsqueeze(-1).expand(-1, -1, self.T_past)
        agent_mask_flat = agent_mask_expanded.contiguous().view(batch_size * num_agents, self.T_past)
        
        # Combine masks: True where padded OR agent doesn't exist
        attention_mask = (obs_mask_flat == 0) | (agent_mask_flat == 0)
        
        # Encode - alternating between temporal and agent attention
        memory = self.transformer_encoder(
            past_emb, 
            batch_size=batch_size, 
            num_agents=num_agents, 
            seq_len=self.T_past,
            attention_mask=attention_mask
        )
        
        # Estimate current positions (only for existing agents)
        current_pos_est = self._estimate_current_position(
            memory, batch['obs_masks'], batch['past_positions'], batch['agent_masks']
        )
        
        # Generate future predictions
        if use_teacher_forcing and self.training:
            predictions = self._teacher_forcing_predictions(
                memory, attention_mask, current_pos_est, batch['future_positions'],
                location_emb, class_emb, batch_size, num_agents, device
            )
        else:
            predictions = self._generate_autoregressive_predictions(
                memory, attention_mask, current_pos_est, class_emb, 
                location_emb, batch_size, num_agents, device
            )
        
        predictions['current_position_estimate'] = current_pos_est
        return predictions

    def _get_safe_embeddings(self, agent_labels, agent_masks):
        """Get embeddings only for existing agents"""
        existing_mask = (agent_labels != -1)
        safe_labels = torch.where(existing_mask, agent_labels, 0)
        class_emb_all = self.class_embedding(safe_labels)
        
        # Zero out embeddings for non-existent agents using agent_masks
        class_emb = class_emb_all * agent_masks.unsqueeze(-1).float()
        return class_emb

    def _generate_autoregressive_predictions(self, memory, attention_mask, current_pos_est,
                                           class_emb, location_emb, 
                                           batch_size, num_agents, device):
        """Generate predictions autoregressively with modified output heads"""
        current_pos = current_pos_est.view(batch_size * num_agents, 2)
        
        # Context embeddings (NO agent embedding)
        class_flat = class_emb.view(batch_size * num_agents, self.d_model)
        location_flat = location_emb.unsqueeze(1).expand(batch_size, num_agents, -1).contiguous()
        location_flat = location_flat.view(batch_size * num_agents, self.d_model)
        context_emb = class_flat + location_flat
        
        # Storage for predictions
        future_deltas_mu = []
        future_deltas_logvar = []
        future_positions = []
        
        for t in range(self.T_future):
            # Create decoder input
            decoder_input = self.pos_projection(current_pos).unsqueeze(1)  # (B*N, 1, d_model)
            decoder_input = decoder_input + context_emb.unsqueeze(1)
            
            # Add temporal encoding
            temporal_pos = self.T_past + t
            if temporal_pos < self.temporal_encoding.pe.size(0):
                temporal_emb = self.temporal_encoding.pe[temporal_pos]  # (d_model,)
                decoder_input = decoder_input + temporal_emb.unsqueeze(0).unsqueeze(0)
            
            # Apply decoder (standard format)
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=None,  # No causal mask needed for single timestep
                memory_key_padding_mask=attention_mask
            )
            
            # Generate predictions with separate heads
            delta_mu = self.delta_head(decoder_output.squeeze(1))  # (B*N, 2)
            log_var = self.logvar_head(decoder_output.squeeze(1))  # (B*N, 2)
            
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
        
        # Propagate variance (convert from log variance)
        delta_var = torch.exp(deltas_logvar)
        positions_var = self._propagate_variance(delta_var)
        
        return {
            'future_deltas_mu': deltas_mu.view(batch_size, num_agents, self.T_future, 2),
            'future_deltas_logvar': deltas_logvar.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_mu': positions.view(batch_size, num_agents, self.T_future, 2),
            'future_positions_var': positions_var.view(batch_size, num_agents, self.T_future, 2)
        }

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
