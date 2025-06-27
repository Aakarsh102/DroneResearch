# import torch
# import torch.nn as nn
# import numpy as np
# import math

# def create_causal_mask(seq_length: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
#     """
#     Creates a causal mask (upper-triangular mask with -inf above the diagonal).
#     This ensures that position i can only attend to positions j where j <= i.
    
#     Args:
#         seq_length (int): The length of the sequence.
#         device (torch.device): Device to place the mask on.
    
#     Returns:
#         torch.Tensor: A (seq_length, seq_length) mask tensor.
#     """
#     mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
#     mask = mask.masked_fill(mask == 1, float('-inf')).to(device)
#     return mask

# class PositionalEncoding(nn.Module):
#     """
#     Sinusoidal positional encoding for transformer models.
#     """
#     def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         # Create positional encoding matrix
#         pe = torch.zeros(max_seq_len, d_model)
#         position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
#                            (-math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # Add batch dimension
        
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape (batch_size, seq_len, d_model)
#         """
#         seq_len = x.size(1)
#         x = x + self.pe[:, :seq_len]
#         return self.dropout(x)

# class DecoderOnlyTrajectoryModel(nn.Module):
#     """
#     Decoder-only transformer model for trajectory prediction with teacher forcing.
    
#     Architecture similar to GPT:
#     - Input: [BOS, x1, y1, x2, y2, ..., xT, yT]  (past trajectory)
#     - Output: [x1, y1, x2, y2, ..., xT, yT, x_{T+1}, y_{T+1}, ..., x_{T+K}, y_{T+K}]
    
#     During training with teacher forcing:
#     - Input: [BOS, past_trajectory, future_trajectory[:-1]]
#     - Target: [past_trajectory, future_trajectory]
    
#     During inference:
#     - Input: [BOS, past_trajectory]
#     - Generate future positions autoregressively
#     """
    
#     def __init__(
#         self, 
#         input_dim: int = 2,           # x, y coordinates
#         hidden_dim: int = 128,        # transformer hidden dimension  
#         num_layers: int = 2,          # number of transformer layers
#         num_heads: int = 8,           # number of attention heads
#         dropout: float = 0.1,         # dropout rate
#         max_seq_len: int = 1000,      # maximum sequence length
#         past_steps: int = 5,          # number of past trajectory steps
#         future_steps: int = 5         # number of future steps to predict
#     ):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.past_steps = past_steps
#         self.future_steps = future_steps
#         self.total_steps = past_steps + future_steps
        
#         # Special tokens
#         self.bos_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
#         # Input embedding - projects (x,y) coordinates to hidden_dim
#         self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
#         # Positional encoding
#         self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
#         # Transformer decoder layers
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 4,
#             dropout=dropout,
#             activation='gelu',
#             batch_first=True,
#             norm_first=True  # Pre-norm architecture (more stable)
#         )
        
#         self.transformer_layers = nn.TransformerDecoder(
#             decoder_layer, 
#             num_layers=num_layers
#         )
        
#         # Output head - projects hidden states back to coordinates
#         self.output_head = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, input_dim)
#         )
        
#         # Initialize weights
#         self._init_weights()
    
#     def _init_weights(self):
#         """Initialize model weights using Xavier initialization."""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     torch.nn.init.zeros_(module.bias)
    
#     def forward(self, past_trajectory, future_trajectory=None):
#         """
#         Forward pass for the decoder-only model.
        
#         Args:
#             past_trajectory: (batch_size, past_steps, input_dim) - observed trajectory
#             future_trajectory: (batch_size, future_steps, input_dim) - target future trajectory
#                               Only provided during training for teacher forcing
        
#         Returns:
#             If training (future_trajectory provided):
#                 - predictions: (batch_size, past_steps + future_steps, input_dim)
#             If inference (future_trajectory is None):
#                 - predictions: (batch_size, future_steps, input_dim)
#         """
#         batch_size = past_trajectory.size(0)
#         device = past_trajectory.device
        
#         if self.training and future_trajectory is not None:
#             # Training mode with teacher forcing
#             return self._forward_teacher_forcing(past_trajectory, future_trajectory)
#         else:
#             # Inference mode - autoregressive generation
#             return self._forward_inference(past_trajectory)
    
#     def _forward_teacher_forcing(self, past_trajectory, future_trajectory):
#         """
#         Training forward pass with teacher forcing.
        
#         Input sequence: [BOS, past_traj, future_traj[:-1]]
#         Target sequence: [past_traj, future_traj]
#         """
#         batch_size = past_trajectory.size(0)
#         device = past_trajectory.device
        
#         # Combine past and future trajectories for teacher forcing
#         # Remove last future step from input (we predict it)
#         full_trajectory = torch.cat([past_trajectory, future_trajectory[:, :-1,:]], dim=1)
        
#         # Embed trajectory points
#         embedded_traj = self.input_embedding(full_trajectory)  # (B, T-1, H)
        
#         # Add BOS token at the beginning
#         bos_tokens = self.bos_token.expand(batch_size, 1, -1)  # (B, 1, H)
#         sequence = torch.cat([bos_tokens, embedded_traj], dim=1)  # (B, T, H)
        
#         # Add positional encoding
#         sequence = self.pos_encoding(sequence)
        
#         # Create causal mask
#         seq_len = sequence.size(1)
#         causal_mask = create_causal_mask(seq_len, device)
        
#         # Pass through transformer (using self as both memory and target)
#         # In decoder-only models, we use the same sequence as tgt and memory
#         output = self.transformer_layers(
#             tgt=sequence,
#             memory=sequence,  # Self-attention
#             tgt_mask=causal_mask
#         )
        
#         # Project to coordinate spaceclear
#         predictions = self.output_head(output)  # (B, T, input_dim)
        
#         # Remove BOS token from output and return predictions
#         print(predictions.shape)
#         return predictions[:, 1:, :]  # (B, past_steps + future_steps, input_dim)
    
#     def _forward_inference(self, past_trajectory):
#         """
#         Inference forward pass with autoregressive generation.
#         """
#         batch_size = past_trajectory.size(0)
#         device = past_trajectory.device
        
#         # Start with past trajectory embedded
#         embedded_past = self.input_embedding(past_trajectory)  # (B, past_steps, H)
        
#         # Add BOS token
#         bos_tokens = self.bos_token.expand(batch_size, 1, -1)
#         sequence = torch.cat([bos_tokens, embedded_past], dim=1)  # (B, past_steps+1, H)
        
#         predictions = []
        
#         # Generate future steps autoregressively
#         for step in range(self.future_steps):
#             # Add positional encoding
#             pos_encoded = self.pos_encoding(sequence)
            
#             # Create causal mask
#             seq_len = sequence.size(1)
#             causal_mask = create_causal_mask(seq_len, device)
            
#             # Forward pass
#             output = self.transformer_layers(
#                 tgt=pos_encoded,
#                 memory=pos_encoded,
#                 tgt_mask=causal_mask
#             )
            
#             # Get prediction for next step
#             next_pos_logits = self.output_head(output[:, -1:, :])  # (B, 1, input_dim)
#             predictions.append(next_pos_logits)
            
#             # Add predicted position to sequence for next iteration
#             next_pos_embedded = self.input_embedding(next_pos_logits)
#             sequence = torch.cat([sequence, next_pos_embedded], dim=1)
        
#         # Concatenate all predictions
#         future_predictions = torch.cat(predictions, dim=1)  # (B, future_steps, input_dim)
#         return future_predictions
    
#     def compute_teacher_forcing_loss(self, past_trajectory, future_trajectory, criterion):
#         """
#         Compute loss using teacher forcing.
        
#         Args:
#             past_trajectory: (batch_size, past_steps, input_dim)
#             future_trajectory: (batch_size, future_steps, input_dim)
#             criterion: Loss function (e.g., nn.MSELoss())
        
#         Returns:
#             loss: Scalar loss value
#         """
#         # Get predictions using teacher forcing
#         predictions = self.forward(past_trajectory, future_trajectory)
        
#         # Create target sequence: [past_trajectory, future_trajectory]
#         target_sequence = torch.cat([past_trajectory, future_trajectory], dim=1)
        
#         # Compute loss
#         # ('H')
#         loss = criterion(predictions, target_sequence)
#         # print('I')
#         return loss

# # Custom dataset class for the decoder-only model
# class DecoderOnlyTrajectoryDataset:
#     """
#     Dataset wrapper that provides the correct input format for decoder-only model.
#     """
    
#     def __init__(self, original_dataset):
#         self.original_dataset = original_dataset
    
#     def __len__(self):
#         return len(self.original_dataset)
    
#     def __getitem__(self, idx):
#         input_seq, target_seq = self.original_dataset[idx]
#         # input_seq: past trajectory (5 steps)
#         # target_seq: future trajectory (5 steps)
#         return input_seq, target_seq

# # Training example
# def train_decoder_only_model():
#     """
#     Example training loop for the decoder-only model.
#     """
#     # Model configuration
#     model = DecoderOnlyTrajectoryModel(
#         input_dim=2,
#         hidden_dim=128,
#         num_layers=6,
#         num_heads=8,
#         dropout=0.1,
#         past_steps=5,
#         future_steps=5
#     )
    
#     # Loss and optimizer
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
#     # Example training step
#     batch_size = 32
#     past_steps = 5
#     future_steps = 5
    
#     # Dummy data
#     past_traj = torch.randn(batch_size, past_steps, 2)
#     future_traj = torch.randn(batch_size, future_steps, 2)
    
#     # Training step
#     model.train()
#     optimizer.zero_grad()
    
#     # Compute loss using teacher forcing
#     loss = model.compute_teacher_forcing_loss(past_traj, future_traj, criterion)
    
#     loss.backward()
    
#     # Gradient clipping for stability
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
#     optimizer.step()
    
#     print(f"Training loss: {loss.item():.6f}")
    
#     # Inference example
#     model.eval()
#     with torch.inference_mode():
#         predictions = model(past_traj)  # Only past trajectory needed
#         print(f"Prediction shape: {predictions.shape}")  # Should be (32, 5, 2)

# # Test the model
# def test_decoder_only_model():
#     """Test the decoder-only model architecture."""
#     print("Testing Decoder-Only Trajectory Model")
#     print("=" * 50)
    
#     # Create model
#     model = DecoderOnlyTrajectoryModel(
#         input_dim=2,
#         hidden_dim=64,  # Smaller for testing
#         num_layers=2,
#         num_heads=4,
#         past_steps=5,
#         future_steps=5
#     )
    
#     batch_size = 8
#     past_traj = torch.randn(batch_size, 5, 2)
#     future_traj = torch.randn(batch_size, 5, 2)
    
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
#     # Test training mode (teacher forcing)
#     model.train()
#     with torch.inference_mode():
#         train_output = model(past_traj, future_traj)
#         print(f"Training output shape: {train_output.shape}")  # Should be (8, 10, 2)
    
#     # Test inference mode
#     model.eval()
#     with torch.inferen():
#         inference_output = model(past_traj)
#         print(f"Inference output shape: {inference_output.shape}")  # Should be (8, 5, 2)
    
#     # Test loss computation
#     criterion = nn.MSELoss()
#     loss = model.compute_teacher_forcing_loss(past_traj, future_traj, criterion)
#     print(f"Example loss: {loss.item():.6f}")
    
#     print("\nModel architecture:")
#     print(model)

# if __name__ == "__main__":
#     test_decoder_only_model()
#     print("\n" + "="*50)
#     train_decoder_only_model()


import torch
import torch.nn as nn
import numpy as np
import math

def create_causal_mask(seq_length: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Creates a causal mask (upper-triangular mask with -inf above the diagonal).
    This ensures that position i can only attend to positions j where j <= i.
    
    Args:
        seq_length (int): The length of the sequence.
        device (torch.device): Device to place the mask on.
    
    Returns:
        torch.Tensor: A (seq_length, seq_length) mask tensor.
    """
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).to(device)
    return mask

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class DecoderOnlyTrajectoryModel(nn.Module):
    """
    Decoder-only transformer model for trajectory prediction with teacher forcing.
    
    Architecture similar to GPT:
    - Input: [BOS, x1, y1, x2, y2, ..., xT, yT]  (past trajectory)
    - Output: [x1, y1, x2, y2, ..., xT, yT, x_{T+1}, y_{T+1}, ..., x_{T+K}, y_{T+K}]
    
    During training with teacher forcing:
    - Input: [BOS, past_trajectory, future_trajectory[:-1]]
    - Target: [past_trajectory, future_trajectory]
    
    During inference:
    - Input: [BOS, past_trajectory]
    - Generate future positions autoregressively
    """
    
    def __init__(
        self, 
        input_dim: int = 2,           # x, y coordinates
        hidden_dim: int = 128,        # transformer hidden dimension  
        num_layers: int = 2,          # number of transformer layers
        num_heads: int = 8,           # number of attention heads
        dropout: float = 0.1,         # dropout rate
        max_seq_len: int = 1000,      # maximum sequence length
        past_steps: int = 5,          # number of past trajectory steps
        future_steps: int = 5         # number of future steps to predict
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.total_steps = past_steps + future_steps
        
        # Special tokens
        self.bos_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Input embedding - projects (x,y) coordinates to hidden_dim
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        
        self.transformer_layers = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # Output head - projects hidden states back to coordinates
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, past_trajectory, future_trajectory=None):
        """
        Forward pass for the decoder-only model.
        
        Args:
            past_trajectory: (batch_size, past_steps, input_dim) - observed trajectory
            future_trajectory: (batch_size, future_steps, input_dim) - target future trajectory
                              Only provided during training for teacher forcing
        
        Returns:
            If training (future_trajectory provided):
                - predictions: (batch_size, past_steps + future_steps, input_dim)
            If inference (future_trajectory is None):
                - predictions: (batch_size, future_steps, input_dim)
        """
        batch_size = past_trajectory.size(0)
        device = past_trajectory.device
        
        if self.training and future_trajectory is not None:
            # Training mode with teacher forcing
            return self._forward_teacher_forcing(past_trajectory, future_trajectory)
        else:
            # Inference mode - autoregressive generation
            return self._forward_inference(past_trajectory)
    
    # def _forward_teacher_forcing(self, past_trajectory, future_trajectory):
    #     """
    #     Training forward pass with teacher forcing.
        
    #     The key insight: we want to predict each position given all previous positions.
    #     Input sequence: [BOS, past_traj, future_traj[:-1]]  
    #     Target sequence: [past_traj, future_traj]
        
    #     So if past=[p1,p2,p3] and future=[f1,f2,f3], then:
    #     Input:  [BOS, p1, p2, p3, f1, f2]  (missing last f3)
    #     Output: [p1, p2, p3, f1, f2, f3]   (shifted by 1)
    #     """
    #     batch_size = past_trajectory.size(0)
    #     device = past_trajectory.device
        
    #     # For teacher forcing, we use past + future[:-1] as input
    #     # This means we're trying to predict future[-1] given everything before it
    #     input_trajectory = torch.cat([past_trajectory, future_trajectory[:, :-1, :]], dim=1)
        
    #     # Embed trajectory points
    #     embedded_traj = self.input_embedding(input_trajectory)  # (B, past_steps + future_steps - 1, H)
        
    #     # Add BOS token at the beginning
    #     bos_tokens = self.bos_token.expand(batch_size, 1, -1)  # (B, 1, H)
    #     sequence = torch.cat([bos_tokens, embedded_traj], dim=1)  # (B, past_steps + future_steps, H)
        
    #     # Add positional encoding
    #     sequence = self.pos_encoding(sequence)
        
    #     # Create causal mask
    #     seq_len = sequence.size(1)
    #     causal_mask = create_causal_mask(seq_len, device)
        
    #     # Pass through transformer layers
    #     output = sequence
    #     for layer in self.transformer_layers.layers:
    #         output = layer(output, output, tgt_mask=causal_mask)
        
    #     # Project to coordinate space
    #     predictions = self.output_head(output)  # (B, past_steps + future_steps, input_dim)
        
    #     # Remove BOS token prediction and return
    #     # This gives us predictions for [past_traj, future_traj]
    #     return predictions[:, 1:, :]  # (B, past_steps + future_steps, input_dim)
    def _forward_teacher_forcing(self, past_trajectory, future_trajectory):
        """
        Training forward pass with teacher forcing.
        
        The key insight: we want to predict each position given all previous positions.
        Input sequence: [BOS, past_traj, future_traj]  
        Target sequence: [past_traj, future_traj]
        
        So if past=[p1,p2,p3] and future=[f1,f2,f3], then:
        Input:  [BOS, p1, p2, p3, f1, f2, f3]  
        Output: [p1, p2, p3, f1, f2, f3]   (shifted by 1 due to BOS)
        """
        batch_size = past_trajectory.size(0)
        device = past_trajectory.device
        
        # For teacher forcing, we use past + future as input
        input_trajectory = torch.cat([past_trajectory, future_trajectory], dim=1)
        
        # Embed trajectory points
        embedded_traj = self.input_embedding(input_trajectory)  # (B, past_steps + future_steps, H)
        
        # Add BOS token at the beginning
        bos_tokens = self.bos_token.expand(batch_size, 1, -1)  # (B, 1, H)
        sequence = torch.cat([bos_tokens, embedded_traj], dim=1)  # (B, past_steps + future_steps + 1, H)
        
        # Add positional encoding
        sequence = self.pos_encoding(sequence)
        
        # Create causal mask
        seq_len = sequence.size(1)
        causal_mask = create_causal_mask(seq_len, device)
        
        # Pass through transformer layers
        output = sequence
        for layer in self.transformer_layers.layers:
            output = layer(output, output, tgt_mask=causal_mask)
        
        # Project to coordinate space
        predictions = self.output_head(output)  # (B, past_steps + future_steps + 1, input_dim)
        
        # Remove BOS token prediction and return
        # This gives us predictions for [past_traj, future_traj]
        return predictions[:, 1:, :]  # (B, past_steps + future_steps, input_dim)
    
    def _forward_inference(self, past_trajectory):
        """
        Inference forward pass with autoregressive generation.
        """
        batch_size = past_trajectory.size(0)
        device = past_trajectory.device
        
        # Start with past trajectory embedded
        embedded_past = self.input_embedding(past_trajectory)  # (B, past_steps, H)
        
        # Add BOS token
        bos_tokens = self.bos_token.expand(batch_size, 1, -1)
        sequence = torch.cat([bos_tokens, embedded_past], dim=1)  # (B, past_steps+1, H)
        
        predictions = []
        
        # Generate future steps autoregressively
        for step in range(self.future_steps):
            # Add positional encoding to current sequence
            pos_encoded = self.pos_encoding(sequence)
            
            # Create causal mask
            seq_len = sequence.size(1)
            causal_mask = create_causal_mask(seq_len, device)
            
            # Forward pass through transformer layers
            output = pos_encoded
            for layer in self.transformer_layers.layers:
                output = layer(output, output, tgt_mask=causal_mask)
            
            # Get prediction for next step
            next_pos_logits = self.output_head(output[:, -1:, :])  # (B, 1, input_dim)
            predictions.append(next_pos_logits)
            
            # Add predicted position to sequence for next iteration
            next_pos_embedded = self.input_embedding(next_pos_logits)
            sequence = torch.cat([sequence, next_pos_embedded], dim=1)
        
        # Concatenate all predictions
        future_predictions = torch.cat(predictions, dim=1)  # (B, future_steps, input_dim)
        return future_predictions
    
    def compute_teacher_forcing_loss(self, past_trajectory, future_trajectory, criterion):
        """
        Compute loss using teacher forcing.
        
        Args:
            past_trajectory: (batch_size, past_steps, input_dim)
            future_trajectory: (batch_size, future_steps, input_dim)
            criterion: Loss function (e.g., nn.MSELoss())
        
        Returns:
            loss: Scalar loss value
        """
        # Get predictions using teacher forcing
        predictions = self._forward_teacher_forcing(past_trajectory, future_trajectory)
        
        # Create target sequence: [past_trajectory, future_trajectory]
        target_sequence = torch.cat([past_trajectory, future_trajectory], dim=1)
        
        # Compute loss
        loss = criterion(predictions, target_sequence)
        return loss

# Custom dataset class for the decoder-only model
class DecoderOnlyTrajectoryDataset:
    """
    Dataset wrapper that provides the correct input format for decoder-only model.
    """
    
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.original_dataset[idx]
        # input_seq: past trajectory (5 steps)
        # target_seq: future trajectory (5 steps)
        return input_seq, target_seq

# Training example
def train_decoder_only_model():
    """
    Example training loop for the decoder-only model.
    """
    # Model configuration
    model = DecoderOnlyTrajectoryModel(
        input_dim=2,
        hidden_dim=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        past_steps=5,
        future_steps=5
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Example training step
    batch_size = 32
    past_steps = 5
    future_steps = 5
    
    # Dummy data
    past_traj = torch.randn(batch_size, past_steps, 2)
    future_traj = torch.randn(batch_size, future_steps, 2)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Compute loss using teacher forcing
    loss = model.compute_teacher_forcing_loss(past_traj, future_traj, criterion)
    
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    print(f"Training loss: {loss.item():.6f}")
    
    # Inference example
    model.eval()
    with torch.inference_mode():
        predictions = model(past_traj)  # Only past trajectory needed
        print(f"Prediction shape: {predictions.shape}")  # Should be (32, 5, 2)

# Test the model
def test_decoder_only_model():
    """Test the decoder-only model architecture."""
    print("Testing Decoder-Only Trajectory Model")
    print("=" * 50)
    
    # Create model
    model = DecoderOnlyTrajectoryModel(
        input_dim=2,
        hidden_dim=64,  # Smaller for testing
        num_layers=2,
        num_heads=4,
        past_steps=5,
        future_steps=5
    )
    
    batch_size = 8
    past_traj = torch.randn(batch_size, 5, 2)
    future_traj = torch.randn(batch_size, 5, 2)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test training mode (teacher forcing)
    model.train()
    with torch.inference_mode():
        train_output = model(past_traj, future_traj)
        print(f"Training output shape: {train_output.shape}")  # Should be (8, 10, 2)
    
    # Test inference mode
    model.eval()
    with torch.inference_mode():
        inference_output = model(past_traj)
        print(f"Inference output shape: {inference_output.shape}")  # Should be (8, 5, 2)
    
    # Test loss computation
    criterion = nn.MSELoss()
    loss = model.compute_teacher_forcing_loss(past_traj, future_traj, criterion)
    print(f"Example loss: {loss.item():.6f}")
    
    print("\nModel architecture:")
    print(model)

if __name__ == "__main__":
    test_decoder_only_model()
    print("\n" + "="*50)
    train_decoder_only_model()