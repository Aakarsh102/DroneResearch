import torch.nn as nn
import torch
import os 
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
  
class LandscapeAwareTrajectoryPredictor(nn.Module):
    def __init__(self, num_classes, locations, d_model=256, nhead=8, num_layers=6, 
                 T_past=10, T_future=10, use_deltas=True, predict_uncertainty=True):
        super(LandscapeAwareTrajectoryPredictor, self).__init__()
        
        self.d_model = d_model
        self.T_past = T_past
        self.T_future = T_future
        self.use_deltas = use_deltas
        self.num_classes = num_classes
        self.predict_uncertainty = predict_uncertainty

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

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, 
                                                  dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Transformer decoder for future prediction
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model*4,
                                                  dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Position interpolation network to estimate current position
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

    def _teacher_forcing_predictions(self, memory, attention_mask, current_pos_est, 
                                    future_positions_target, location_emb, class_emb, device):
        """
        Teacher forcing for training
        
        Args:
            memory: Encoded past sequence
            attention_mask: Mask for past sequence
            current_pos_est: Current position estimate (batch, 2)
            future_positions_target: Ground truth future positions (batch, T_future, 2)
            location_emb, class_emb: Context embeddings
            device: Device
        """
        batch_size = current_pos_est.size(0)
        
        # Create decoder input sequence: [current_pos, future_pos[0], ..., future_pos[T_future-2]]
        # We don't include the last future position because we predict it
        decoder_positions = torch.cat([
            current_pos_est.unsqueeze(1),  # (batch, 1, 2)
            future_positions_target[:, :-1, :]  # (batch, T_future-1, 2)
        ], dim=1)  # (batch, T_future, 2)
        
        # Project positions to embedding dimension
        decoder_input = self.pos_projection(decoder_positions)  # (batch, T_future, d_model)
        
        # Add context embeddings (location and class)
        context_emb = location_emb + class_emb  # (batch, d_model)
        decoder_input = decoder_input + context_emb.unsqueeze(1)  # (batch, T_future, d_model)
        
        # Add temporal encoding for future timesteps
        for i in range(self.T_future):
            temporal_pos = self.T_past + i
            if temporal_pos < self.temporal_encoding.pe.size(0):
                temporal_emb = self.temporal_encoding.pe[temporal_pos]  # (1, d_model)
                decoder_input[:, i, :] += temporal_emb
        
        # Create causal mask for autoregressive property
        causal_mask = torch.triu(torch.ones(self.T_future, self.T_future, device=device) * float('-inf'), diagonal=1)
        
        # Apply transformer decoder with teacher forcing
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,  # (batch, T_future, d_model)
            memory=memory,      # (batch, T_past, d_model)
            tgt_mask=causal_mask,  # (T_future, T_future)
            memory_key_padding_mask=attention_mask  # (batch, T_past)
        )  # (batch, T_future, d_model)
        
        # Generate predictions for each timestep
        if self.use_deltas:
            delta_predictions = self.delta_head(decoder_output)  # (batch, T_future, 2 or 4)
            
            if self.predict_uncertainty:
                delta_mu = delta_predictions[..., :2]
                delta_logvar = delta_predictions[..., 2:]
                
                # Convert deltas to positions
                positions = self._convert_deltas_to_positions_teacher_forcing(
                    current_pos_est, delta_mu
                )
                future_positions_logvar = self._propagate_uncertainty(delta_logvar)
                
                return {
                    'future_deltas_mu': delta_mu,
                    'future_deltas_logvar': delta_logvar,
                    'future_positions_mu': positions,
                    'future_positions_logvar': future_positions_logvar 
                }
            else:
                delta_pred = delta_predictions
                positions = self._convert_deltas_to_positions_teacher_forcing(
                    current_pos_est, delta_pred
                )
                
                return {
                    'future_deltas': delta_pred,
                    'future_positions': positions
                }
        else:
            # Direct position predictions
            if self.predict_uncertainty:
                pos_predictions = self.position_head(decoder_output)
                return {
                    'future_positions_mu': pos_predictions[..., :2],
                    'future_positions_logvar': pos_predictions[..., 2:]
                }
            else:
                return {
                    'future_positions': self.position_head(decoder_output)
                }

    def _convert_deltas_to_positions_teacher_forcing(self, start_pos, deltas):
        """Convert delta predictions to positions during teacher forcing"""
        batch_size, seq_len, _ = deltas.shape
        positions = torch.zeros_like(deltas)
    
        current_pos = start_pos
        for i in range(seq_len):
            current_pos = current_pos + deltas[:, i, :]
            positions[:, i, :] = current_pos

        return positions

# Modified forward method to use teacher forcing during training
    def forward(self, batch, use_teacher_forcing=True):
        batch_size = batch['past_positions'].size(0)
        device = batch['past_positions'].device

        location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], device=device)
        location_emb = self.location_embedding(location_indices) 
        class_emb = self.class_embedding(batch['label'])

        past_positions = batch['past_positions']
        past_positions_orig = batch['past_positions_orig']
        obs_mask = batch['obs_mask']

        # Create enhanced past embeddings
        past_emb = self._create_enhanced_past_embeddings(
            past_positions, past_positions_orig, obs_mask, 
            location_emb, class_emb, device
        )
        
        attention_mask = (obs_mask == 0)
        memory = self.transformer_encoder(past_emb, src_key_padding_mask=attention_mask)
        
        # Estimate current position
        current_pos_est = self._estimate_current_position(memory, obs_mask, past_positions)
        
        # Use teacher forcing during training, autoregressive during inference
        if use_teacher_forcing and self.training:
            future_predictions = self._teacher_forcing_predictions(
                memory, attention_mask, current_pos_est, 
                batch['future_positions'], location_emb, class_emb, device
            )
        else:
            future_predictions = self._generate_autoregressive_predictions(
                memory, attention_mask, current_pos_est, location_emb, class_emb, device
            )
        
        # Process outputs
        if use_teacher_forcing and self.training:
            outputs = future_predictions  # Already in correct format
        else:
            outputs = self._process_autoregressive_outputs(future_predictions, current_pos_est)
        
        outputs['current_position_estimate'] = current_pos_est
        return outputs
    
    def _create_enhanced_past_embeddings(self, past_positions, past_positions_orig, 
                                       obs_mask, location_emb, class_emb, device):
        """Create enhanced embeddings that include gap information"""
        batch_size = past_positions.size(0)
        

        gaps = torch.zeros_like(obs_mask, dtype=torch.long)
        for i in range(batch_size):
            gap_counter = 0
            for j in range(self.T_past):
                if obs_mask[i, j] == 1:
                    gap_counter = 0
                else:
                    gap_counter += 1
                gaps[i, j] = min(gap_counter, self.T_past)  # Cap at T_past
        

        pos_emb = torch.zeros(batch_size, self.T_past, self.d_model, device=device)
        
        for i in range(batch_size):
            for j in range(self.T_past):
                if obs_mask[i, j] == 1:

                    pos_emb[i, j] = self.pos_projection(past_positions[i, j])
                else:
                    # Unobserved position - use learned token
                    pos_emb[i, j] = self.unobserved_token
        

        spatial_emb = self.spatial_encoding(past_positions_orig)
        obs_mask_expanded = obs_mask.unsqueeze(-1).expand(-1, -1, self.d_model)
        pos_emb = pos_emb + spatial_emb * obs_mask_expanded

        gap_emb = self.gap_encoding(gaps)
        pos_emb = pos_emb + gap_emb
        

        pos_emb = pos_emb + location_emb.unsqueeze(1) + class_emb.unsqueeze(1)
        

        pos_emb = self.temporal_encoding(pos_emb.transpose(0, 1)).transpose(0, 1)
        
        return pos_emb
    
    def _estimate_current_position(self, memory, obs_mask, past_positions):
        """Estimate current position using all available information"""
        batch_size = memory.size(0)
        
        # Use attention-weighted combination of all memory states
        # Focus more on recent observations
        weights = torch.softmax(memory.mean(dim=-1), dim=-1)  # (batch, T_past)
        weights = weights * obs_mask  # Zero out unobserved positions
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
        
        # Weighted combination of memory states
        context = torch.sum(memory * weights.unsqueeze(-1), dim=1)  # (batch, d_model)
        
        # Predict current position
        current_pos = self.position_interpolator(context)  # (batch, 2)
        
        return current_pos
    
    def _prepare_initial_decoder_input(self, current_pos_est, device):
        """Prepare initial decoder input (just the starting position)"""
        batch_size = current_pos_est.size(0)
        
        # Start with current position estimate - just the first timestep
        decoder_input = self.pos_projection(current_pos_est)  # (batch, d_model)
        decoder_input = decoder_input.unsqueeze(1)  # (batch, 1, d_model)
        
        return decoder_input
    
        
    def _generate_autoregressive_predictions(self, memory, attention_mask, 
                                        current_pos_est, location_emb, class_emb, device):
        """Generate future predictions autoregressively"""
        batch_size = current_pos_est.size(0)
        
        # Initialize outputs
        future_positions = []
        future_deltas = []
        future_uncertainties = []
        
        # Start with current position
        current_pos = current_pos_est  # (batch_size, 2)
        
        # Context embeddings
        context_emb = location_emb + class_emb  # (batch_size, d_model)
        
        for t in range(self.T_future):
            # Prepare decoder input for current timestep
            # Project current position to embedding space
            decoder_input = self.pos_projection(current_pos)  # (batch_size, d_model)
            decoder_input = decoder_input.unsqueeze(1)  # (batch_size, 1, d_model)
            
            # Add context embeddings
            decoder_input = decoder_input + context_emb.unsqueeze(1)  # (batch_size, 1, d_model)
            
            # Add temporal encoding for current future timestep
            temporal_pos = self.T_past + t
            if temporal_pos < self.temporal_encoding.pe.size(0):
                # Get temporal embedding - handle the positional encoding correctly
                temporal_emb = self.temporal_encoding.pe[temporal_pos:temporal_pos+1]  # (1, d_model)
                temporal_emb = temporal_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
                decoder_input = decoder_input + temporal_emb
            
            # Apply transformer decoder - NO TRANSPOSING needed since batch_first=True
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,  # (batch_size, 1, d_model)
                memory=memory,      # (batch_size, T_past, d_model)
                memory_key_padding_mask=attention_mask  # (batch_size, T_past)
            )  # (batch_size, 1, d_model)
            
            # Squeeze to remove sequence dimension
            decoder_output = decoder_output.squeeze(1)  # (batch_size, d_model)
            
            # Predict next position/delta
            if self.use_deltas:
                delta_pred = self.delta_head(decoder_output)  # (batch_size, 2 or 4)
                
                if self.predict_uncertainty:
                    delta_mu = delta_pred[..., :2]  # (batch_size, 2)
                    delta_logvar = delta_pred[..., 2:]  # (batch_size, 2)
                    
                    # Use mean for next position calculation
                    next_pos = current_pos + delta_mu
                    
                    future_deltas.append(delta_pred)
                    future_uncertainties.append(delta_logvar)
                else:
                    delta_mu = delta_pred
                    next_pos = current_pos + delta_mu
                    future_deltas.append(delta_pred)
                
                future_positions.append(next_pos)
                current_pos = next_pos
            else:
                pos_pred = self.position_head(decoder_output)  # (batch_size, 2 or 4)
                
                if self.predict_uncertainty:
                    pos_mu = pos_pred[..., :2]
                    pos_logvar = pos_pred[..., 2:]
                    
                    future_positions.append(pos_pred)
                    future_uncertainties.append(pos_logvar)
                    current_pos = pos_mu  # Use mean for next step
                else:
                    future_positions.append(pos_pred)
                    current_pos = pos_pred
        
        return {
            'positions': future_positions,
            'deltas': future_deltas,
            'uncertainties': future_uncertainties
        }
    
    def _process_autoregressive_outputs(self, predictions, current_pos_est):
        """Process autoregressive predictions into final output format"""
        outputs = {}
        
        if predictions['positions']:
            # Stack predictions
            if self.predict_uncertainty:
                if self.use_deltas:
                    # Deltas with uncertainty
                    delta_stack = torch.stack(predictions['deltas'], dim=1)  # (batch, T_future, 4)
                    outputs['future_deltas_mu'] = delta_stack[..., :2]
                    outputs['future_deltas_logvar'] = delta_stack[..., 2:]
                    
                    # Positions (already computed during autoregressive generation)
                    pos_stack = torch.stack(predictions['positions'], dim=1)  # (batch, T_future, 2)
                    outputs['future_positions_mu'] = pos_stack
                    
                    # Uncertainty propagation for positions
                    outputs['future_positions_logvar'] = self._propagate_uncertainty(
                        outputs['future_deltas_logvar']
                    )
                else:
                    # Direct positions with uncertainty
                    pos_stack = torch.stack(predictions['positions'], dim=1)  # (batch, T_future, 4)
                    outputs['future_positions_mu'] = pos_stack[..., :2]
                    outputs['future_positions_logvar'] = pos_stack[..., 2:]
            else:
                if self.use_deltas:
                    # Deltas without uncertainty
                    outputs['future_deltas'] = torch.stack(predictions['deltas'], dim=1)
                    outputs['future_positions'] = torch.stack(predictions['positions'], dim=1)
                else:
                    # Direct positions without uncertainty
                    outputs['future_positions'] = torch.stack(predictions['positions'], dim=1)
        
        return outputs
    
    def _propagate_uncertainty(self, delta_logvar):
        """Propagate uncertainty through cumulative sum"""
        batch_size, seq_len, _ = delta_logvar.shape
        
        # Convert log variance to variance
        delta_var = torch.exp(delta_logvar)
        
        # Cumulative uncertainty propagation
        pos_var = torch.zeros_like(delta_var)
        pos_var[:, 0] = delta_var[:, 0]
        
        for i in range(1, seq_len):
            # Uncertainty accumulates when integrating deltas
            pos_var[:, i] = pos_var[:, i-1] + delta_var[:, i]
        
        # Convert back to log variance
        return torch.log(pos_var + 1e-8)
    
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
    
    def predict_autoregressive(self, batch, sampling_strategy='mean', temperature=1.0):
        """
        Inference-time autoregressive prediction with sampling options
        
        Args:
            batch: Input batch
            sampling_strategy: 'mean', 'sample', or 'top_k'
            temperature: Temperature for sampling (only used if sampling_strategy != 'mean')
        """
        self.eval()
        with torch.no_grad():
            batch_size = batch['past_positions'].size(0)
            device = batch['past_positions'].device

            # Encode past sequence (same as training)
            location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], device=device)
            location_emb = self.location_embedding(location_indices) 
            class_emb = self.class_embedding(batch['label'])
            
            past_positions = batch['past_positions']
            past_positions_orig = batch['past_positions_orig']
            obs_mask = batch['obs_mask']
            
            # Create enhanced past embeddings
            past_emb = self._create_enhanced_past_embeddings(
                past_positions, past_positions_orig, obs_mask, 
                location_emb, class_emb, device
            )
            
            attention_mask = (obs_mask == 0)
            memory = self.transformer_encoder(past_emb, src_key_padding_mask=attention_mask)
            
            # Estimate current position
            current_pos_est = self._estimate_current_position(memory, obs_mask, past_positions)
            
            # Generate predictions with specified sampling strategy
            if sampling_strategy == 'mean':
                predictions = self._generate_autoregressive_predictions(
                    memory, attention_mask, current_pos_est, location_emb, class_emb, device
                )
            else:
                predictions = self._generate_autoregressive_predictions_with_sampling(
                    memory, attention_mask, current_pos_est, location_emb, class_emb, 
                    device, sampling_strategy, temperature
                )
            
            outputs = self._process_autoregressive_outputs(predictions, current_pos_est)
            outputs['current_position_estimate'] = current_pos_est
            
            return outputs
    
    def _generate_autoregressive_predictions_with_sampling(self, memory, attention_mask, 
                                                         current_pos_est, location_emb, class_emb, 
                                                         device, sampling_strategy, temperature):
        """Generate predictions with uncertainty sampling"""
        batch_size = current_pos_est.size(0)
        
        future_positions = []
        future_deltas = []
        future_uncertainties = []
        
        current_pos = current_pos_est
        decoder_input = self._prepare_initial_decoder_input(current_pos, device)
        context_emb = location_emb + class_emb
        decoder_input = decoder_input + context_emb.unsqueeze(1)
        
        for t in range(self.T_future):
            temporal_pos = self.T_past + t
            if temporal_pos < self.temporal_encoding.pe.size(0):
                temporal_emb = self.temporal_encoding.pe[temporal_pos].unsqueeze(0).unsqueeze(0)
                decoder_input_t = decoder_input + temporal_emb
            else:
                decoder_input_t = decoder_input
            
            decoder_output = self.transformer_decoder(
                decoder_input_t, memory, 
                memory_key_padding_mask=attention_mask
            )
            
            if self.use_deltas:
                delta_pred = self.delta_head(decoder_output.squeeze(1))
                
                if self.predict_uncertainty:
                    delta_mu = delta_pred[..., :2]
                    delta_logvar = delta_pred[..., 2:]
                    
                    # Sample from the distribution
                    if sampling_strategy == 'sample':
                        delta_std = torch.exp(0.5 * delta_logvar)
                        eps = torch.randn_like(delta_mu)
                        delta_sampled = delta_mu + temperature * delta_std * eps
                    else:  # mean
                        delta_sampled = delta_mu
                    
                    next_pos = current_pos + delta_sampled
                    future_deltas.append(delta_pred)
                    future_uncertainties.append(delta_logvar)
                else:
                    delta_sampled = delta_pred
                    next_pos = current_pos + delta_sampled
                    future_deltas.append(delta_pred)
                
                future_positions.append(next_pos)
                current_pos = next_pos
            else:
                pos_pred = self.position_head(decoder_output.squeeze(1))
                
                if self.predict_uncertainty:
                    pos_mu = pos_pred[..., :2]
                    pos_logvar = pos_pred[..., 2:]
                    
                    if sampling_strategy == 'sample':
                        pos_std = torch.exp(0.5 * pos_logvar)
                        eps = torch.randn_like(pos_mu)
                        pos_sampled = pos_mu + temperature * pos_std * eps
                    else:
                        pos_sampled = pos_mu
                    
                    future_positions.append(pos_pred)
                    future_uncertainties.append(pos_logvar)
                    current_pos = pos_sampled
                else:
                    future_positions.append(pos_pred)
                    current_pos = pos_pred
            
            # Update decoder input for next timestep
            if t < self.T_future - 1:
                decoder_input = self.pos_projection(current_pos).unsqueeze(1)
                decoder_input = decoder_input + context_emb.unsqueeze(1)
        
        return {
            'positions': future_positions,
            'deltas': future_deltas,
            'uncertainties': future_uncertainties
        }


# Evaluation metrics
def compute_metrics(predictions, targets, video_stats=None):
    """Compute evaluation metrics"""
    metrics = {}
    
    # Get predictions and targets
    if 'future_positions_mu' in predictions:
        pred_pos = predictions['future_positions_mu']
        pred_uncertainty = torch.exp(predictions['future_positions_logvar'])
    else:
        pred_pos = predictions['future_positions']
        pred_uncertainty = None
    
    target_pos = targets['future_positions']
    valid_mask = targets['occ_mask']
    
    # Denormalize if needed
    if video_stats is not None:
        pred_pos = denormalize_positions(pred_pos, video_stats)
        target_pos = denormalize_positions(target_pos, video_stats)
    
    # Compute ADE (Average Displacement Error)
    displacement = torch.norm(pred_pos - target_pos, dim=-1)  # (batch, T_future)
    masked_displacement = displacement * valid_mask
    ade = masked_displacement.sum() / valid_mask.sum()
    metrics['ADE'] = ade.item()
    
    # Compute FDE (Final Displacement Error)
    final_displacement = displacement[:, -1]  # Last timestep
    final_valid = valid_mask[:, -1]
    fde = (final_displacement * final_valid).sum() / final_valid.sum()
    metrics['FDE'] = fde.item()
    
    # Uncertainty calibration metrics
    if pred_uncertainty is not None:
        # Prediction interval coverage
        std = torch.sqrt(pred_uncertainty)
        lower_bound = pred_pos - 1.96 * std  # 95% confidence interval
        upper_bound = pred_pos + 1.96 * std
        
        within_interval = ((target_pos >= lower_bound) & (target_pos <= upper_bound)).float()
        coverage = (within_interval * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum()
        metrics['Coverage_95'] = coverage.item()
        
        # Average uncertainty
        avg_uncertainty = (pred_uncertainty * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum()
        metrics['Avg_Uncertainty'] = avg_uncertainty.item()
    
    return metrics




"""
Issues: 
1. only training with atleast 1 observed, this will screw things up. (Make it 2 atleast) (what about unobserved) (done)
2. Implement teacher forcing (done)
3. do you have to implement delta logvar 
4. do you have to normalize (might drown out the input signal due to so many embeddings) (won't now)
5. **do you need the unobserved token since during inference that isn't used at all. You'll have incomplete sequences.
and you'll want to continue it from there. **
6. Is the estimate_current_position considering when there's no unobserved location.
7. attention mask for now  
"""

"""
For the graph model, 
1. The location embedding should be added after the message passing 
2. The spatial encoding too should be added after 
3. 
"""