import torch.nn as nn
import torch
import os 
from prototype1 import SpatialEncoding, PositionalEncoding, denormalize_positions




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
                temporal_emb = self.temporal_encoding.pe[temporal_pos].unsqueeze(0)  # (1, d_model)
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
                
                return {
                    'future_deltas_mu': delta_mu,
                    'future_deltas_logvar': delta_logvar,
                    'future_positions_mu': positions
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



        
    # def forward(self, batch):
    #     batch_size = batch['past_positions'].size(0)
    #     device = batch['past_positions'].device

    #     location_indices = torch.tensor([self.location_to_idx[loc] for loc in batch['location']], device=device)
    #     location_emb = self.location_embedding(location_indices) 

    #     class_emb = self.class_embedding(batch['label'])

    #     past_positions = batch['past_positions']  # (batch, T_past, 2) - normalized
    #     past_positions_orig = batch['past_positions_orig']  # (batch, T_past, 2) - original scale
    #     obs_mask = batch['obs_mask']  # (batch, T_past)

    #     # Create enhanced past embeddings with gap information
    #     past_emb = self._create_enhanced_past_embeddings(
    #         past_positions, past_positions_orig, obs_mask, 
    #         location_emb, class_emb, device
    #     )
        
    #     # Create attention mask for observed positions
    #     attention_mask = (obs_mask == 0)  # True for unobserved positions
        
    #     # Encode past sequence
    #     memory = self.transformer_encoder(past_emb, src_key_padding_mask=attention_mask)
        
    #     # Estimate current position using all available information
    #     current_pos_est = self._estimate_current_position(memory, obs_mask, past_positions)
        
    #     # Generate future predictions autoregressively
    #     future_predictions = self._generate_autoregressive_predictions(
    #         memory, attention_mask, current_pos_est, location_emb, class_emb, device
    #     )
        
    #     # Output predictions
    #     outputs = self._process_autoregressive_outputs(future_predictions, current_pos_est)
        
    #     # Add current position estimate to outputs for evaluation
    #     outputs['current_position_estimate'] = current_pos_est
        
    #     return outputs
    
    def _create_enhanced_past_embeddings(self, past_positions, past_positions_orig, 
                                       obs_mask, location_emb, class_emb, device):
        """Create enhanced embeddings that include gap information"""
        batch_size = past_positions.size(0)
        
        # Calculate temporal gaps (frames since last observation)
        gaps = torch.zeros_like(obs_mask, dtype=torch.long)
        for i in range(batch_size):
            gap_counter = 0
            for j in range(self.T_past):
                if obs_mask[i, j] == 1:
                    gap_counter = 0
                else:
                    gap_counter += 1
                gaps[i, j] = min(gap_counter, self.T_past)  # Cap at T_past
        
        # Create position embeddings
        pos_emb = torch.zeros(batch_size, self.T_past, self.d_model, device=device)
        
        for i in range(batch_size):
            for j in range(self.T_past):
                if obs_mask[i, j] == 1:
                    # Observed position
                    pos_emb[i, j] = self.pos_projection(past_positions[i, j])
                else:
                    # Unobserved position - use learned token
                    pos_emb[i, j] = self.unobserved_token
        
        # Add spatial encoding (only for observed positions)
        spatial_emb = self.spatial_encoding(past_positions_orig)
        obs_mask_expanded = obs_mask.unsqueeze(-1).expand(-1, -1, self.d_model)
        pos_emb = pos_emb + spatial_emb * obs_mask_expanded
        
        # Add gap encoding
        gap_emb = self.gap_encoding(gaps)
        pos_emb = pos_emb + gap_emb
        
        # Add location and class context
        pos_emb = pos_emb + location_emb.unsqueeze(1) + class_emb.unsqueeze(1)
        
        # Add temporal encoding
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
        current_pos = current_pos_est
        
        # Initialize decoder input with current position
        decoder_input = self._prepare_initial_decoder_input(current_pos, device)
        
        # Add context embeddings (location and class) - this is appropriate here
        context_emb = location_emb + class_emb
        decoder_input = decoder_input + context_emb.unsqueeze(1)
        
        for t in range(self.T_future):
            # Add temporal encoding for current timestep
            temporal_pos = self.T_past + t
            if temporal_pos < self.temporal_encoding.pe.size(0):
                temporal_emb = self.temporal_encoding.pe[temporal_pos].unsqueeze(0).unsqueeze(0)
                decoder_input_t = decoder_input + temporal_emb
            else:
                decoder_input_t = decoder_input
            
            # Decode one step
            decoder_output = self.transformer_decoder(
                decoder_input_t, memory, 
                memory_key_padding_mask=attention_mask
            )  # (batch, 1, d_model)
            
            # Predict next position/delta
            if self.use_deltas:
                delta_pred = self.delta_head(decoder_output.squeeze(1))  # (batch, 2 or 4)
                
                if self.predict_uncertainty:
                    delta_mu = delta_pred[..., :2]
                    delta_logvar = delta_pred[..., 2:]
                    
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
                pos_pred = self.position_head(decoder_output.squeeze(1))  # (batch, 2 or 4)
                
                if self.predict_uncertainty:
                    pos_mu = pos_pred[..., :2]
                    pos_logvar = pos_pred[..., 2:]
                    
                    future_positions.append(pos_pred)
                    future_uncertainties.append(pos_logvar)
                    current_pos = pos_mu  # Use mean for next step
                else:
                    future_positions.append(pos_pred)
                    current_pos = pos_pred
            
            # Update decoder input for next timestep
            if t < self.T_future - 1:  # Don't update on last iteration
                decoder_input = self.pos_projection(current_pos).unsqueeze(1)
                decoder_input = decoder_input + context_emb.unsqueeze(1)
        
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


class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction with uncertainty"""
    
    def __init__(self, use_deltas=True, predict_uncertainty=True, 
                 position_weight=1.0, delta_weight=1.0, uncertainty_weight=0.1):
        super().__init__()
        self.use_deltas = use_deltas
        self.predict_uncertainty = predict_uncertainty
        self.position_weight = position_weight
        self.delta_weight = delta_weight
        self.uncertainty_weight = uncertainty_weight
        
    def forward(self, predictions, targets):
        total_loss = 0.0
        loss_dict = {}
        
        # Mask for valid (non-occluded) future positions
        valid_mask = targets['occ_mask']  # (batch, T_future)
        
        if self.predict_uncertainty:
            # Negative log likelihood loss for positions
            if 'future_positions_mu' in predictions:
                pos_mu = predictions['future_positions_mu']
                pos_logvar = predictions['future_positions_logvar']
                pos_target = targets['future_positions']
                
                pos_loss = self._gaussian_nll_loss(pos_mu, pos_logvar, pos_target, valid_mask)
                loss_dict['position_loss'] = pos_loss
                total_loss += self.position_weight * pos_loss
            
            # Negative log likelihood loss for deltas
            if self.use_deltas and 'future_deltas_mu' in predictions:
                delta_mu = predictions['future_deltas_mu']
                delta_logvar = predictions['future_deltas_logvar']
                delta_target = targets['future_deltas']
                
                delta_loss = self._gaussian_nll_loss(delta_mu, delta_logvar, delta_target, valid_mask)
                loss_dict['delta_loss'] = delta_loss
                total_loss += self.delta_weight * delta_loss
            
            # Uncertainty regularization (prevent overconfident predictions)
            if 'future_positions_logvar' in predictions:
                uncertainty_reg = -predictions['future_positions_logvar'].mean()
                loss_dict['uncertainty_reg'] = uncertainty_reg
                total_loss += self.uncertainty_weight * uncertainty_reg
                
        else:
            # Standard MSE loss
            if 'future_positions' in predictions:
                pos_loss = self._masked_mse_loss(
                    predictions['future_positions'], targets['future_positions'], valid_mask
                )
                loss_dict['position_loss'] = pos_loss
                total_loss += self.position_weight * pos_loss
            
            if self.use_deltas and 'future_deltas' in predictions:
                delta_loss = self._masked_mse_loss(
                    predictions['future_deltas'], targets['future_deltas'], valid_mask
                )
                loss_dict['delta_loss'] = delta_loss
                total_loss += self.delta_weight * delta_loss
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
    
    def _gaussian_nll_loss(self, mu, logvar, target, mask):
        """Negative log likelihood for Gaussian distribution"""
        # Expand mask to match tensor dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(mu)
        
        # Compute NLL only for valid positions
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + ((target - mu) ** 2) / var)
        
        # Apply mask and average
        masked_nll = nll * mask_expanded
        return masked_nll.sum() / mask_expanded.sum()
    
    def _masked_mse_loss(self, pred, target, mask):
        """MSE loss with masking"""
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)
        mse = ((pred - target) ** 2) * mask_expanded
        return mse.sum() / mask_expanded.sum()


# Training function example
def train_step(model, batch, loss_fn, optimizer):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    predictions = model(batch)
    loss, loss_dict = loss_fn(predictions, batch)
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), loss_dict


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
"""