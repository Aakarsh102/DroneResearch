import torch
import torch.nn as nn
import numpy as np

def create_causal_mask(seq_length: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Creates a causal mask (upper-triangular mask with -inf above the diagonal).
    
    Args:
        seq_length (int): The length of the sequence.
        device (torch.device): Device to place the mask on (CPU or GPU).
    
    Returns:
        torch.Tensor: A (seq_length, seq_length) mask tensor suitable for tgt_mask.
    """
    # Create upper triangular matrix with 1s above diagonal
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    # Replace 1s with -inf to mask those positions
    mask = mask.masked_fill(mask == 1, float('-inf')).to(device)
    return mask

class PositionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, max_seq_len: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create the div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float) * 
                           (-np.log(10000.0) / hidden_dim))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, hidden_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Apply positional encoding to input tensor"""
        # x shape: (batch_size, seq_len, hidden_dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class ImprovedTrajectoryModel(nn.Module):
    """
    Transformer-based trajectory prediction model with proper BOS token handling.
    
    This model uses an encoder-decoder architecture where:
    - Encoder processes the input trajectory sequence
    - Decoder generates future trajectory points autoregressively
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, 
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 nhead: int = 8, dropout: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Input embedding layers
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Special tokens
        # BOS token: learnable parameter that indicates start of sequence
        self.bos_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional encoding
        self.position_encoder = PositionEncoder(hidden_dim, max_seq_len)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier handling
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, src, tgt_len: int = 5):
        """
        Forward pass for trajectory prediction.
        
        Args:
            src: Input trajectory sequence (batch_size, src_len, input_dim)
            tgt_len: Number of future steps to predict
            
        Returns:
            Predicted trajectory points (batch_size, tgt_len, input_dim)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode input sequence
        src_embedded = self.input_embedding(src)
        src_embedded = self.position_encoder(src_embedded)
        src_embedded = self.layer_norm(src_embedded)
        
        # Prepare decoder input with BOS token
        # Start with BOS token for the decoder
        bos_tokens = self.bos_token.expand(batch_size, 1, -1)  # (batch_size, 1, hidden_dim)
        
        # For training, we'll use teacher forcing
        # For inference, we'll generate autoregressively
        if self.training:
            # Teacher forcing: use ground truth for faster training
            return self._forward_teacher_forcing(src_embedded, bos_tokens, tgt_len)
        else:
            # Autoregressive generation for inference
            return self._forward_autoregressive(src_embedded, bos_tokens, tgt_len)
    
    def _forward_teacher_forcing(self, src_embedded, bos_tokens, tgt_len):
        """Forward pass with teacher forcing (for training)"""
        batch_size = src_embedded.size(0)
        device = src_embedded.device
        
        # Create target sequence starting with BOS
        # In teacher forcing, we'd typically have the target sequence
        # For now, we'll generate step by step
        tgt_sequence = bos_tokens
        predictions = []
        
        for step in range(tgt_len):
            # Apply positional encoding to target
            tgt_pos_encoded = self.position_encoder(tgt_sequence)
            tgt_pos_encoded = self.layer_norm(tgt_pos_encoded)
            
            # Create causal mask for target sequence
            tgt_mask = create_causal_mask(tgt_sequence.size(1), device)
            
            # Transformer forward pass
            output = self.transformer(
                src=src_embedded,
                tgt=tgt_pos_encoded,
                tgt_mask=tgt_mask
            )
            
            # Get the last output and project to coordinate space
            last_output = output[:, -1:, :]  # (batch_size, 1, hidden_dim)
            predicted_pos = self.output_projection(last_output)  # (batch_size, 1, input_dim)
            
            predictions.append(predicted_pos)
            
            # For next iteration, add the predicted position (embedded) to target sequence
            predicted_embedded = self.input_embedding(predicted_pos)
            tgt_sequence = torch.cat([tgt_sequence, predicted_embedded], dim=1)
        
        # Concatenate all predictions
        return torch.cat(predictions, dim=1)  # (batch_size, tgt_len, input_dim)
    
    def _forward_autoregressive(self, src_embedded, bos_tokens, tgt_len):
        """Forward pass with autoregressive generation (for inference)"""
        batch_size = src_embedded.size(0)
        device = src_embedded.device
        
        tgt_sequence = bos_tokens
        predictions = []
        
        for step in range(tgt_len):
            # Apply positional encoding
            tgt_pos_encoded = self.position_encoder(tgt_sequence)
            tgt_pos_encoded = self.layer_norm(tgt_pos_encoded)
            
            # Create causal mask
            tgt_mask = create_causal_mask(tgt_sequence.size(1), device)
            
            # Transformer forward pass
            with torch.no_grad() if not self.training else torch.enable_grad():
                output = self.transformer(
                    src=src_embedded,
                    tgt=tgt_pos_encoded,
                    tgt_mask=tgt_mask
                )
            
            # Get prediction for next position
            last_output = output[:, -1:, :]
            predicted_pos = self.output_projection(last_output)
            
            predictions.append(predicted_pos)
            
            # Add predicted position to target sequence for next step
            predicted_embedded = self.input_embedding(predicted_pos)
            tgt_sequence = torch.cat([tgt_sequence, predicted_embedded], dim=1)
        
        return torch.cat(predictions, dim=1)

# Alternative simpler approach using encoder-only architecture
class SimpleTransformerTrajectoryModel(nn.Module):
    """
    Simpler transformer model that directly predicts all future positions.
    This might be easier to train and debug.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, 
                 num_layers: int = 2, nhead: int = 8, 
                 output_steps: int = 5, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_steps = output_steps
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.position_encoder = PositionEncoder(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Pool sequence to single vector
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_steps * input_dim)
        )
        
    def forward(self, x):
        """
        Forward pass that directly predicts all future positions.
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            
        Returns:
            Predicted positions (batch_size, output_steps, input_dim)
        """
        batch_size = x.size(0)
        
        # Embed and add positional encoding
        x_embedded = self.input_embedding(x)
        x_embedded = self.position_encoder(x_embedded)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(x_embedded)
        
        # Global pooling to get fixed-size representation
        # encoded shape: (batch_size, seq_len, hidden_dim)
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)  # (batch_size, hidden_dim)
        
        # Generate all predictions at once
        predictions = self.output_mlp(pooled)  # (batch_size, output_steps * input_dim)
        
        # Reshape to desired output format
        return predictions.view(batch_size, self.output_steps, self.input_dim)

# Usage example and testing
def test_models():
    """Test both model architectures"""
    batch_size = 8
    seq_len = 5
    input_dim = 2
    output_steps = 5
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("Testing ImprovedTrajectoryModel (Encoder-Decoder):")
    model1 = ImprovedTrajectoryModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        nhead=4
    )
    
    # Test forward pass
    model1.eval()
    with torch.no_grad():
        output1 = model1(x, tgt_len=output_steps)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output1.shape}")
        print(f"Output sample: {output1[0, :2, :]}")
    
    print("\nTesting SimpleTransformerTrajectoryModel (Encoder-only):")
    model2 = SimpleTransformerTrajectoryModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        nhead=4,
        output_steps=output_steps
    )
    
    output2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Output sample: {output2[0, :2, :]}")

if __name__ == "__main__":
    test_models()