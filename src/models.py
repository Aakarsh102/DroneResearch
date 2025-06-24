import torch
import torch.nn as nn


class TrajectoryPredictionModel(nn.Module):
    def __init__(self, sequence_input_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(sequence_input_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last timestep

class CompressionModel(nn.Module):
    def __init__(self, input_dim: int, compressed_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, compressed_dim)
    
    def forward(self, camera_observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        # Simple concatenation and compression for now
        all_obs = []
        for cam_id, obs in camera_observations.items():
            if len(obs) > 0:
                all_obs.append(obs.mean(dim=0))  # Average observations per camera
        if all_obs:
            combined = torch.stack(all_obs).mean(dim=0)
            return self.encoder(combined)
        return torch.zeros(128)  # Empty observation

class GlobalIntegrationModel(nn.Module):
    def __init__(self, local_pred_dim: int, compressed_dim: int, output_dim: int, total_objects: int):
        super().__init__()
        self.total_objects = total_objects
        self.fc = nn.Linear(local_pred_dim + compressed_dim, output_dim * total_objects)
    
    def forward(self, local_predictions: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        # Broadcast global context to match local predictions
        batch_size = local_predictions.size(0)
        global_expanded = global_context.unsqueeze(0).expand(batch_size, -1)
        
        # Concatenate and predict for all objects
        combined = torch.cat([local_predictions.flatten(1), global_expanded], dim=1)
        output = self.fc(combined)
        return output.view(batch_size, self.total_objects, -1)
