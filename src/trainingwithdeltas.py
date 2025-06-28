import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
from tqdm import tqdm
import logging

# Delta-based Decoder-Only Transformer Model
class DecoderOnlyTrajectoryModel(nn.Module):
    """Delta-based decoder-only transformer for trajectory prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 num_heads: int, dropout: float, past_steps: int, future_steps: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.past_steps = past_steps
        self.future_steps = future_steps
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(past_steps + future_steps, hidden_dim))
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to predict delta values
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def convert_to_deltas(self, positions: torch.Tensor) -> torch.Tensor:
        """Convert absolute positions to delta values."""
        # positions: (B, seq_len, 2)
        deltas = torch.zeros_like(positions)
        deltas[:, 1:] = positions[:, 1:] - positions[:, :-1]  # dx, dy
        deltas[:, 0] = 0  # First delta is zero (no previous position)
        return deltas
    
    def convert_deltas_to_positions(self, deltas: torch.Tensor, initial_pos: torch.Tensor) -> torch.Tensor:
        """Convert delta values back to absolute positions."""
        # deltas: (B, seq_len, 2)
        # initial_pos: (B, 2) - last known position
        positions = torch.zeros_like(deltas)
        positions[:, 0] = initial_pos
        
        for i in range(1, deltas.size(1)):
            positions[:, i] = positions[:, i-1] + deltas[:, i]
        
        return positions
    
    def forward(self, past_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference (autoregressive generation).
        
        Args:
            past_trajectory: (B, past_steps, 2) - absolute coordinates
            
        Returns:
            future_trajectory: (B, future_steps, 2) - absolute coordinates
        """
        batch_size = past_trajectory.size(0)
        device = past_trajectory.device
        
        # Convert past trajectory to deltas
        past_deltas = self.convert_to_deltas(past_trajectory)
        
        # Project input deltas
        past_embeddings = self.input_projection(past_deltas)  # (B, past_steps, hidden_dim)
        
        # Add positional encoding for past
        past_embeddings += self.pos_embedding[:self.past_steps].unsqueeze(0)
        past_embeddings = self.dropout(self.layer_norm(past_embeddings))
        
        # Initialize future predictions with zeros
        future_deltas = torch.zeros(batch_size, self.future_steps, self.input_dim, device=device)
        future_positions = torch.zeros(batch_size, self.future_steps, self.input_dim, device=device)
        
        # Current position (last position from past trajectory)
        current_pos = past_trajectory[:, -1]  # (B, 2)
        
        # Autoregressive generation
        for step in range(self.future_steps):
            # Combine past deltas with predicted future deltas up to current step
            if step == 0:
                combined_deltas = past_deltas
            else:
                combined_deltas = torch.cat([past_deltas, future_deltas[:, :step]], dim=1)
            
            # Project combined sequence
            combined_embeddings = self.input_projection(combined_deltas)
            
            # Add positional encoding
            seq_len = combined_embeddings.size(1)
            combined_embeddings += self.pos_embedding[:seq_len].unsqueeze(0)
            combined_embeddings = self.dropout(self.layer_norm(combined_embeddings))
            
            # Create causal mask
            causal_mask = self.create_causal_mask(seq_len).to(device)
            
            # Apply transformer
            output = self.transformer(combined_embeddings, combined_embeddings, 
                                    tgt_mask=causal_mask, memory_mask=causal_mask)
            
            # Predict next delta
            next_delta = self.output_projection(output[:, -1])  # (B, 2)
            future_deltas[:, step] = next_delta
            
            # Convert to absolute position
            current_pos = current_pos + next_delta
            future_positions[:, step] = current_pos
        
        return future_positions
    
    def compute_teacher_forcing_loss(self, past_trajectory: torch.Tensor, 
                                   future_trajectory: torch.Tensor, 
                                   criterion: nn.Module) -> torch.Tensor:
        """
        Compute loss using teacher forcing during training.
        
        Args:
            past_trajectory: (B, past_steps, 2) - absolute coordinates
            future_trajectory: (B, future_steps, 2) - absolute coordinates
            criterion: loss function
            
        Returns:
            loss: scalar tensor
        """
        batch_size = past_trajectory.size(0)
        device = past_trajectory.device
        
        # Convert to deltas
        past_deltas = self.convert_to_deltas(past_trajectory)
        future_deltas = self.convert_to_deltas(
            torch.cat([past_trajectory[:, -1:], future_trajectory], dim=1)
        )[:, 1:]  # Remove first element (which would be zero)
        
        # Combine past and future for teacher forcing
        full_sequence = torch.cat([past_trajectory, future_trajectory], dim=1)  # (B, total_steps, 2)
        full_deltas = self.convert_to_deltas(full_sequence)
        
        # Project to embeddings
        embeddings = self.input_projection(full_deltas)
        
        # Add positional encoding
        total_steps = embeddings.size(1)
        embeddings += self.pos_embedding[:total_steps].unsqueeze(0)
        embeddings = self.dropout(self.layer_norm(embeddings))
        
        # Create causal mask
        causal_mask = self.create_causal_mask(total_steps).to(device)
        
        # Apply transformer
        output = self.transformer(embeddings, embeddings, 
                                tgt_mask=causal_mask, memory_mask=causal_mask)
        
        # Predict deltas
        predicted_deltas = self.output_projection(output)
        
        # We want to predict future deltas given past + current context
        # Target: future deltas
        # Prediction: output deltas corresponding to future positions
        target_deltas = full_deltas[:, self.past_steps:]  # Future deltas
        predicted_future_deltas = predicted_deltas[:, self.past_steps-1:-1]  # Shift by 1 for prediction
        
        # Compute loss on delta predictions
        loss = criterion(predicted_future_deltas, target_deltas)
        
        return loss


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


@dataclass
class DeltaDecoderTrainingConfig:
    sequence_length: int = 5        # past trajectory steps
    prediction_steps: int = 5       # future trajectory steps  
    batch_size: int = 32           # smaller batch size for transformer
    learning_rate: float = 1e-4    # lower learning rate for transformer
    num_epochs: int = 50
    hidden_dim: int = 128
    num_layers: int = 6            # transformer layers
    num_heads: int = 8             # attention heads
    dropout: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    save_model_path: str = 'delta_decoder_trajectory_model.pth'
    log_interval: int = 10
    early_stopping_patience: int = 10
    min_delta: float = 0.001
    weight_decay: float = 0.01     # L2 regularization
    warmup_steps: int = 1000       # learning rate warmup


class DeltaDecoderTrajectoryTrainer:
    """Training class specifically for delta-based decoder-only transformer model."""
    
    def __init__(self, config: DeltaDecoderTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize delta-based decoder-only model
        self.model = DecoderOnlyTrajectoryModel(
            input_dim=2,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            past_steps=config.sequence_length,
            future_steps=config.prediction_steps
        ).to(self.device)
        
        # Try to compile model if available
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, backend='inductor', mode='reduce-overhead')
                print("Model compiled using torch.compile")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        
        # AdamW optimizer (better for transformers)
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Cosine annealing scheduler with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.min_delta
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Training step counter for warmup
        self.training_step = 0
    
    def apply_warmup(self):
        """Apply learning rate warmup for the first few steps."""
        if self.training_step < self.config.warmup_steps:
            warmup_lr = self.config.learning_rate * (self.training_step / self.config.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        self.training_step += 1
    
    def prepare_data(self, global_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders."""
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
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.logger.info(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch using teacher forcing with delta predictions."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc='Training (Delta-based)')
        
        for batch_inputs, batch_targets in progress_bar:
            # batch_inputs: (B, seq_len, 2) - past trajectory (absolute coordinates)
            # batch_targets: (B, pred_steps, 2) - future trajectory (absolute coordinates)
            
            batch_inputs = batch_inputs.to(self.device, non_blocking=True)
            batch_targets = batch_targets.to(self.device, non_blocking=True)
            
            # Apply learning rate warmup
            self.apply_warmup()
            
            # Forward pass with teacher forcing
            self.optimizer.zero_grad()
            
            # The model handles delta conversion internally
            loss = self.model.compute_teacher_forcing_loss(
                batch_inputs, batch_targets, self.criterion
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.inference_mode():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                # Compute validation loss using teacher forcing
                loss = self.model.compute_teacher_forcing_loss(
                    batch_inputs, batch_targets, self.criterion
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def test_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test the model using inference mode (no teacher forcing)."""
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        
        mae_criterion = nn.L1Loss()
        
        with torch.inference_mode():
            for batch_inputs, batch_targets in test_loader:
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                # Use inference mode (no teacher forcing)
                predictions = self.model(batch_inputs)  # Only past trajectory needed
                
                # Compute metrics
                mse_loss = self.criterion(predictions, batch_targets)
                mae_loss = mae_criterion(predictions, batch_targets)
                
                batch_size = batch_inputs.size(0)
                total_mse += mse_loss.item() * batch_size
                total_mae += mae_loss.item() * batch_size
                total_samples += batch_size
        
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        rmse = np.sqrt(avg_mse)
        
        return {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': rmse
        }
    
    def train(self, global_df: pd.DataFrame) -> Dict[str, float]:
        """Main training loop."""
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
            
            # Learning rate scheduling (after warmup)
            if self.training_step >= self.config.warmup_steps:
                self.scheduler.step()
            
            # Logging
            if epoch % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f'Epoch {epoch:3d}/{self.config.num_epochs} | '
                    f'Train Loss: {train_loss:.6f} | '
                    f'Val Loss: {val_loss:.6f} | '
                    f'LR: {current_lr:.2e}'
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
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_step': self.training_step
        }
        torch.save(checkpoint, self.config.save_model_path)
    
    def load_model(self):
        """Load model checkpoint."""
        if os.path.exists(self.config.save_model_path):
            checkpoint = torch.load(self.config.save_model_path, 
                                  map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.training_step = checkpoint.get('training_step', 0)
            self.logger.info(f"Model loaded from {self.config.save_model_path}")
        else:
            self.logger.warning(f"No checkpoint found at {self.config.save_model_path}")
    
    def plot_training_history(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Delta-based Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Delta-based Training and Validation Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('delta_decoder_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_predictions(self, global_df: pd.DataFrame, num_samples: int = 5):
        """Visualize prediction examples using inference mode."""
        # Create test dataset
        test_dataset = TrajectoryDataset(
            global_df, 
            self.config.sequence_length, 
            self.config.prediction_steps
        )
        
        # Select random samples
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        self.model.eval()
        plt.figure(figsize=(15, 3 * num_samples))
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                input_seq, target_seq = test_dataset[idx]
                input_seq = input_seq.unsqueeze(0).to(self.device)
                target_seq = target_seq.numpy()
                
                # Use inference mode (autoregressive generation with delta predictions)
                prediction = self.model(input_seq).cpu().numpy().squeeze()
                input_seq = input_seq.cpu().numpy().squeeze()
                
                print(f"Sample {i+1}:")
                print(f"  Input shape: {input_seq.shape}")
                print(f"  Target shape: {target_seq.shape}")
                print(f"  Prediction shape: {prediction.shape}")
                
                plt.subplot(num_samples, 1, i + 1)
                
                # Plot input sequence (past trajectory)
                plt.plot(input_seq[:, 0], input_seq[:, 1], 'bo-', 
                        label='Input (Past)', markersize=6, linewidth=2)
                
                # Plot ground truth future
                plt.plot(target_seq[:, 0], target_seq[:, 1], 'go-', 
                        label='Ground Truth (Future)', markersize=6, linewidth=2)
                
                # Plot prediction
                plt.plot(prediction[:, 0], prediction[:, 1], 'ro-', 
                        label='Delta-based Prediction (Future)', markersize=6, linewidth=2)
                
                # Connect past to future with dotted lines
                plt.plot([input_seq[-1, 0], target_seq[0, 0]], 
                        [input_seq[-1, 1], target_seq[0, 1]], 'g--', 
                        alpha=0.5, label='_nolegend_')
                plt.plot([input_seq[-1, 0], prediction[0, 0]], 
                        [input_seq[-1, 1], prediction[0, 1]], 'r--', 
                        alpha=0.5, label='_nolegend_')
                
                plt.legend()
                plt.title(f'Delta-based Decoder Trajectory Prediction Example {i+1}')
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')  # Equal aspect ratio for better visualization
        
        plt.tight_layout()
        plt.savefig('delta_decoder_prediction_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_delta_predictions(self, global_df: pd.DataFrame, num_samples: int = 3):
        """Visualize delta predictions specifically."""
        test_dataset = TrajectoryDataset(
            global_df, 
            self.config.sequence_length, 
            self.config.prediction_steps
        )
        
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        self.model.eval()
        plt.figure(figsize=(15, 4 * num_samples))
        
        with torch.inference_mode():
            for i, idx in enumerate(indices):
                input_seq, target_seq = test_dataset[idx]
                input_seq_tensor = input_seq.unsqueeze(0).to(self.device)
                
                # Get predictions
                prediction = self.model(input_seq_tensor).cpu().numpy().squeeze()
                input_seq = input_seq.numpy()
                target_seq = target_seq.numpy()
                
                # Convert to deltas for visualization
                input_deltas = np.zeros_like(input_seq)
                input_deltas[1:] = input_seq[1:] - input_seq[:-1]
                
                target_deltas = np.zeros_like(target_seq)
                target_deltas[0] = target_seq[0] - input_seq[-1]
                target_deltas[1:] = target_seq[1:] - target_seq[:-1]
                
                pred_deltas = np.zeros_like(prediction)
                pred_deltas[0] = prediction[0] - input_seq[-1]
                pred_deltas[1:] = prediction[1:] - prediction[:-1]
# Plot position trajectories
                plt.subplot(num_samples, 3, i * 3 + 1)
                plt.plot(input_seq[:, 0], input_seq[:, 1], 'bo-', label='Input (Past)', markersize=4)
                plt.plot(target_seq[:, 0], target_seq[:, 1], 'go-', label='Ground Truth', markersize=4)
                plt.plot(prediction[:, 0], prediction[:, 1], 'ro-', label='Prediction', markersize=4)
                plt.title(f'Positions - Sample {i+1}')
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                
                # Plot X deltas
                plt.subplot(num_samples, 3, i * 3 + 2)
                time_steps_input = range(len(input_deltas))
                time_steps_target = range(len(input_deltas), len(input_deltas) + len(target_deltas))
                time_steps_pred = range(len(input_deltas), len(input_deltas) + len(pred_deltas))
                
                plt.plot(time_steps_input, input_deltas[:, 0], 'bo-', label='Input ΔX', markersize=4)
                plt.plot(time_steps_target, target_deltas[:, 0], 'go-', label='True ΔX', markersize=4)
                plt.plot(time_steps_pred, pred_deltas[:, 0], 'ro-', label='Pred ΔX', markersize=4)
                plt.title(f'X Deltas - Sample {i+1}')
                plt.xlabel('Time Step')
                plt.ylabel('ΔX')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot Y deltas
                plt.subplot(num_samples, 3, i * 3 + 3)
                plt.plot(time_steps_input, input_deltas[:, 1], 'bo-', label='Input ΔY', markersize=4)
                plt.plot(time_steps_target, target_deltas[:, 1], 'go-', label='True ΔY', markersize=4)
                plt.plot(time_steps_pred, pred_deltas[:, 1], 'ro-', label='Pred ΔY', markersize=4)
                plt.title(f'Y Deltas - Sample {i+1}')
                plt.xlabel('Time Step')
                plt.ylabel('ΔY')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('delta_decoder_delta_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_attention_weights(self, global_df: pd.DataFrame, num_samples: int = 3):
        """Analyze attention weights for interpretability."""
        # This method assumes your decoder model has a way to return attention weights
        # You might need to modify your DecoderOnlyTrajectoryModel to support this
        
        test_dataset = TrajectoryDataset(
            global_df, 
            self.config.sequence_length, 
            self.config.prediction_steps
        )
        
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        self.model.eval()
        plt.figure(figsize=(12, 4 * num_samples))
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                input_seq, _ = test_dataset[idx]
                input_seq = input_seq.unsqueeze(0).to(self.device)
                
                # Forward pass (you may need to modify your model to return attention weights)
                try:
                    # This assumes your model has a method to get attention weights
                    _, attention_weights = self.model.forward_with_attention(input_seq)
                    
                    # Plot attention heatmap
                    plt.subplot(num_samples, 1, i + 1)
                    
                    # Average attention across heads and layers
                    avg_attention = attention_weights.mean(dim=1).mean(dim=0).cpu().numpy()
                    
                    im = plt.imshow(avg_attention, cmap='Blues', aspect='auto')
                    plt.colorbar(im)
                    plt.title(f'Attention Weights - Sample {i+1}')
                    plt.xlabel('Key Position')
                    plt.ylabel('Query Position')
                    
                except AttributeError:
                    print(f"Attention analysis not available - model doesn't support attention weight extraction")
                    return
        
        plt.tight_layout()
        plt.savefig('decoder_attention_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


# Main training script for decoder-only model
if __name__ == "__main__":
    # Configuration for decoder-only model
    decoder_config = DeltaDecoderTrainingConfig(
        sequence_length=5,
        prediction_steps=5,
        batch_size=32,              # Smaller batch size for transformer
        learning_rate=1e-4,         # Lower learning rate
        num_epochs=50,
        hidden_dim=128,
        num_layers=6,               # More layers for transformer
        num_heads=8,
        dropout=0.1,
        weight_decay=0.01,
        warmup_steps=1000,
        save_model_path='best_decoder_trajectory_model.pth'
    )
    
    
    # Load data
    print("Loading training data...")
    global_df = pd.read_csv('orbit_simulation_1000_objects_positions.csv')[['object_id', 'time_step', 'x', 'y']]
    
    print(f"Loaded dataset with {len(global_df)} observations")
    print(f"Objects: {global_df['object_id'].nunique()}")
    print(f"Time steps: {global_df['time_step'].nunique()}")
    
    # Initialize decoder trainer
    decoder_trainer = DeltaDecoderTrajectoryTrainer(decoder_config)
    
    # Train model
    print("Starting decoder-only model training...")
    test_metrics = decoder_trainer.train(global_df)
    
    # Plot training history
    decoder_trainer.plot_training_history()
    
    # Visualize predictions
    decoder_trainer.visualize_predictions(global_df, num_samples=5)
    
    # Visualize delta predictions
    decoder_trainer.visualize_delta_predictions(global_df, num_samples=3)
    
    # Analyze attention (if supported by your model)
    # decoder_trainer.analyze_attention_weights(global_df, num_samples=3)
    
    print("Decoder-only training completed successfully!")
    print(f"Final test metrics: {test_metrics}")