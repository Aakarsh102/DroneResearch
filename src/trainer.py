import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
from tqdm import tqdm
import logging

def create_causal_mask(output_length: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Creates a causal mask (upper-triangular mask with -inf above the diagonal).
    
    Args:
        output_length (int): The length of the target/output sequence.
        device (torch.device): Device to place the mask on (CPU or GPU).
    
    Returns:
        torch.Tensor: A (output_length, output_length) mask tensor suitable for tgt_mask.
    """
    # 1s above the diagonal → masked
    mask = torch.triu(torch.ones(output_length, output_length), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).to(device)
    return mask


@dataclass
class TrainingConfig:
    sequence_length: int = 5        # t - number of input time steps
    prediction_steps: int = 5       # t - number of future steps to predict
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 50
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    save_model_path: str = 'trajectory_model.pth'
    log_interval: int = 10
    early_stopping_patience: int = 7
    min_delta: float = 0.001

class TrajectoryDataset(Dataset):
    """Dataset for trajectory prediction training"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int, prediction_steps: int):
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.data = data.sort_values(['object_id', 'time_step'])
        
        # Group by object_id
        self.grouped_data = self.data.groupby('object_id')
        
        # Create sequences
        self.sequences = []
        self._create_sequences()
        
        print(f"Created {len(self.sequences)} training sequences")
    
    def _create_sequences(self):
        """Create input-output sequence pairs for training"""
        for object_id, group in self.grouped_data:
            positions = group[['x', 'y']].values
            time_steps = group['time_step'].values
            
            # Ensure positions are sorted by time
            sorted_indices = np.argsort(time_steps)
            positions = positions[sorted_indices]
            time_steps = time_steps[sorted_indices]
            
            # Create sequences with sliding window
            for i in range(len(positions) - self.sequence_length - self.prediction_steps + 1):
                # Check if time steps are consecutive
                input_times = time_steps[i:i + self.sequence_length]
                output_times = time_steps[i + self.sequence_length:i + self.sequence_length + self.prediction_steps]
                
                # Verify consecutive time steps
                if (np.diff(input_times) == 1).all() and (np.diff(output_times) == 1).all():
                    if input_times[-1] + 1 == output_times[0]:  # Ensure continuity
                        input_seq = positions[i:i + self.sequence_length]
                        output_seq = positions[i + self.sequence_length:i + self.sequence_length + self.prediction_steps]
                        
                        self.sequences.append({
                            'input': torch.FloatTensor(input_seq),
                            'output': torch.FloatTensor(output_seq),
                            'object_id': object_id
                        })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]['input'], self.sequences[idx]['output']

class PositionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, seq_len: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pe = torch.zeros(seq_len, hidden_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float) * (-np.log(10000.0)/ hidden_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe [:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        self.regiester_buffer('pe', self.pe)

    def forward(self, x):
        """Apply positional encoding to input tensor"""
        # x shape: (batch_size, seq_len, hidden_dim)
        seq = x.size(1)
        return x + self.pe[:, :seq]
    
class ImprovedTrajectoryModel(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, fnn_dim: int = 128, output_len:int = 10, num_layers:int = 2):
        super().__init__()
        self.embed = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        # Hardcoded sequence length for simplicity here
        self.position_encoder = PositionEncoder(hidden_dim=hidden_dim, seq_len=5)
        mask = create_causal_mask(output_len, 'cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer = nn.TransformerDecoderLayer(d_model = hidden_dim, nhead=8, dim_feedforward=
                                                      fnn_dim, dropout=0.1, activation='relu', tgt_mask=mask)
        
        self.transformer_decoder = nn.TransformerDecoder(self.transformer, num_layers=num_layers)
        self.fc_out = nn.Linear(in_features=hidden_dim, out_features=output_len * input_dim)

    def forward(self, x):
        x = self.embed(x)  # x shape: (batch_size, seq_len, input_dim)
        x = self.position_encoder(x)  # Apply positional encoding
        x = x.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, hidden_dim) for Transformer
        x = self.transformer_decoder(x, x)  # Transformer decoder expects two inputs
# class ImprovedTrajectoryModel(nn.Module):
#     """Improved trajectory prediction model with better architecture"""
    
#     def __init__(self, input_dim: int = 2, hidden_dim: int = 128, 
#                  output_steps: int = 10, num_layers: int = 2, dropout: float = 0.1):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_steps = output_steps
#         self.num_layers = num_layers
        
#         # LSTM layers with dropout
#         self.lstm = nn.LSTM(
#             input_dim, hidden_dim, num_layers, 
#             batch_first=True, dropout=dropout if num_layers > 1 else 0
#         )
        
#         # Attention mechanism
#         self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
#         # Feed-forward layers
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, output_steps * input_dim)
        
#         # Activation functions
#         self.relu = nn.ReLU()
#         self.layer_norm = nn.LayerNorm(hidden_dim)
        
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, 2)
#         batch_size = x.size(0)
        
#         # LSTM forward pass
#         lstm_out, (hidden, cell) = self.lstm(x)
        
#         # Apply attention
#         attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
#         attended_out = self.layer_norm(attended_out + lstm_out)  # Residual connection
        
#         # Use the last time step output
#         last_output = attended_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
#         # Feed-forward layers
#         out = self.fc1(last_output)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)
#         out = self.relu(out)
        
#         # Output layer
#         predictions = self.output_layer(out)
        
#         # Reshape to (batch_size, output_steps, input_dim)
#         return predictions.view(batch_size, self.output_steps, self.input_dim)

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

class TrajectoryTrainer:
    """Main training class for trajectory prediction model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = ImprovedTrajectoryModel(
            input_dim=2,
            hidden_dim=config.hidden_dim,
            output_steps=config.prediction_steps,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.5)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.min_delta
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self, global_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test data loaders"""
        # for i in global_df:
        #     print(i)
            
        # Create dataset
        full_dataset = TrajectoryDataset(
            global_df, 
            self.config.sequence_length, 
            self.config.prediction_steps
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.validation_split * total_size)
        test_size = total_size - train_size - val_size
        # for i in full_dataset:
            # print(i)
            # break
        train_dataset = torch.tensor([sample for sample in full_dataset if sample[0].item() <= 80])
        test_dataset = torch.tensor([sample for sample in full_dataset if ((sample[0].item() > 80) & (sample[0].item() <= 90))])
        val_dataset = torch.tensor([sample for sample in full_dataset if sample[0].item() > 90])
        # test_dataset = full_dataset[(full_dataset[1] > 80) & (full_dataset[1] <= 90)]
        # val_dataset = full_dataset[full_dataset[1] > 90]
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        self.logger.info(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_inputs, batch_targets in progress_bar:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_inputs)
            loss = self.criterion(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.inference_mode():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_inputs)
                loss = self.criterion(predictions, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def test_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test the model and compute various metrics"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        
        mae_criterion = nn.L1Loss()
        
        with torch.no_grad():
            for batch_inputs, batch_targets in test_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_inputs)
                
                # MSE Loss
                mse_loss = self.criterion(predictions, batch_targets)
                total_loss += mse_loss.item() * batch_inputs.size(0)
                
                # MAE Loss
                mae_loss = mae_criterion(predictions, batch_targets)
                total_mae += mae_loss.item() * batch_inputs.size(0)
                
                total_samples += batch_inputs.size(0)
        
        avg_mse = total_loss / total_samples
        avg_mae = total_mae / total_samples
        rmse = np.sqrt(avg_mse)
        
        return {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': rmse
        }
    
    def train(self, global_df: pd.DataFrame) -> Dict[str, float]:
        """Main training loop"""
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(global_df)
        
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self.logger.info(
                    f'Epoch {epoch:3d}/{self.config.num_epochs} | '
                    f'Train Loss: {train_loss:.6f} | '
                    f'Val Loss: {val_loss:.6f} | '
                    f'LR: {self.optimizer.param_groups[0]["lr"]:.8f}'
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                self.logger.info(f'New best model saved with val loss: {val_loss:.6f}')
            
            # Early stopping
            if self.early_stopping(val_loss):
                self.logger.info(f'Early stopping at epoch {epoch}')
                break
        
        # Final testing
        self.load_model()  # Load best model
        test_metrics = self.test_model(test_loader)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Test Metrics - MSE: {test_metrics['mse']:.6f}, "
                        f"MAE: {test_metrics['mae']:.6f}, RMSE: {test_metrics['rmse']:.6f}")
        
        return test_metrics
    
    def save_model(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, self.config.save_model_path)
    
    def load_model(self):
        """Load model checkpoint"""
        if os.path.exists(self.config.save_model_path):
            checkpoint = torch.load(self.config.save_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.logger.info(f"Model loaded from {self.config.save_model_path}")
        else:
            self.logger.warning(f"No checkpoint found at {self.config.save_model_path}")
    
    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Training and Validation Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_predictions(self, global_df: pd.DataFrame, num_samples: int = 5):
        """Visualize some prediction examples"""
        # Create a small test dataset
        test_dataset = TrajectoryDataset(
            global_df, 
            self.config.sequence_length, 
            self.config.prediction_steps
        )
        
        # Select random samples
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        self.model.eval()
        plt.figure(figsize=(15, 3 * num_samples))
        
        with torch.inference_mode():
            for i, idx in enumerate(indices):
                input_seq, target_seq = test_dataset[idx]
                input_seq = input_seq.unsqueeze(0).to(self.device)
                target_seq = target_seq.numpy()
                
                prediction = self.model(input_seq).cpu().numpy().squeeze()
                input_seq = input_seq.cpu().numpy().squeeze()
                print("input_seq:", input_seq)
                print("target_seq:", target_seq)
                print("prediction:", prediction)
                
                plt.subplot(num_samples, 1, i + 1)
                
                # Plot input sequence
                plt.plot(input_seq[:, 0], input_seq[:, 1], 'bo-', label='Input', markersize=6)
                
                # Plot ground truth future
                plt.plot(target_seq[:, 0], target_seq[:, 1], 'go-', label='Ground Truth', markersize=6)
                
                # Plot prediction
                plt.plot(prediction[:, 0], prediction[:, 1], 'ro-', label='Prediction', markersize=6)
                
                # Connect input to future
                plt.plot([input_seq[-1, 0], target_seq[0, 0]], 
                        [input_seq[-1, 1], target_seq[0, 1]], 'g--', alpha=0.5)
                plt.plot([input_seq[-1, 0], prediction[0, 0]], 
                        [input_seq[-1, 1], prediction[0, 1]], 'r--', alpha=0.5)
                
                plt.legend()
                plt.title(f'Trajectory Prediction Example {i+1}')
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_examples.png', dpi=300, bbox_inches='tight')
        plt.show()



# Main training script
if __name__ == "__main__":
    # Configuration
    config = TrainingConfig(
        sequence_length=5,
        prediction_steps=5,
        batch_size=64,
        learning_rate=0.001,
        num_epochs=100,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        save_model_path='best_trajectory_model.pth'
    )
    
    # # Create training data
    # print("Creating training data...")
    global_df = pd.read_csv('orbit_simulation_1000_objects_positions.csv')[['object_id', 'time_step', 'x', 'y']]
    # # global_df = global_df[1:]
     
    # print(f"Created dataset with {len(global_df)} observations")
    # print(f"Objects: {global_df['object_id'].nunique()}")
    # print(f"Time steps: {global_df['time_step'].nunique()}")
    
    # # Initialize trainer
    trainer = TrajectoryTrainer(config)
    
    # # Train model
    # test_metrics = trainer.train(global_df)
    
    # Plot training history
    # trainer.plot_training_history()
    
    # Visualize predictions
    trainer.visualize_predictions(global_df, num_samples=5)
    
    print("Training completed successfully!")
    #print(f"Final test metrics: {test_metrics}")