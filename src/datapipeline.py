import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

class PipelineConfig:
    sequence_length: int = 10
    input_length: int = 30     # in_len
    output_length: int = 20    # l - in_len
    sync_interval: int = 5     # t - synchronization every t time steps
    batch_size: int = 32
    num_cameras: int = 4
    input_features: int = 6    # time_step, object_id, x, y, radius, angular_velocity, direction
    position_features: int = 2  # x, y for output
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class CameraDataset(Dataset):
    def __init__ (self, df: pd.DataFrame, config: PipelineConfig, camera_id: int):
        self.df = df.sort_values(by = ['object_id', 'time_step'])
        self.config = config
        self.data = self.preprocess_data()
        self.camera_id = camera_id

        self.object_groups = {}
        self.sequence_indices = []
        
        # Group by object_id and store start indices for each valid sequence
        for object_id, group in self.df.groupby('object_id'):
            group_sorted = group.sort_values('time_step').reset_index(drop=True)
            self.object_groups[object_id] = group_sorted
            
            # Calculate valid sequence start positions
            num_sequences = len(group_sorted) - self.config.sequence_length + 1
            for i in range(num_sequences):
                self.sequence_indices.append((object_id, i))

    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        """Generate sequence on-the-fly"""
        object_id, start_idx = self.sequence_indices[idx]
        group_data = self.object_groups[object_id]
        
        # Extract sequence data without copying
        sequence_data = group_data.iloc[start_idx:start_idx + self.config.sequence_length]
        
        # Input features: [time_step, object_id, x, y, radius, angular_velocity, direction]
        input_features = sequence_data[['time_step', 'object_id', 'x', 'y', 'radius', 'angular_velocity', 'direction']].values
        input_seq = torch.FloatTensor(input_features[:self.config.input_length])
        
        # Output targets: [x, y] positions for prediction
        output_positions = sequence_data[['x', 'y']].values[self.config.input_length:]
        output_seq = torch.FloatTensor(output_positions)
        
        # Time steps for synchronization
        time_steps = sequence_data['time_step'].tolist()
        
        return input_seq, output_seq, object_id, time_steps
    
class CameraProcessor:
    def __init__(self, camera_id: int, df: pd.DataFrame, config: PipelineConfig):
        self.camera_id = camera_id
        self.config = config
        self.dataset = CameraDataset(df, config, camera_id)
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.process_data)
        self.thread.start()

        self.trajectory_model = TrajectoryPredictionModel(
            sequence_input_dim=config.input_features,
            output_dim=config.position_features
        )
        self.outgoing_queue = queue.Queue()  # Send compressed data to sync manager
        self.incoming_queue = queue.Queue()  # Receive global context from sync manager

        self.current_time_step = 0
        self.local_predictions = {}

    def extract_time_step_observations(self, time_step: int) -> Dict[int, np.ndarray]:
        """Extract observations for a specific time step."""
        mask = self.df['time_step'] == time_step
        time_step_data = self.df[mask]
        if (len(time_step_data) > 0 ):
            obs = time_step_data[['object_id', 'x', 'y', 'radius', 'angular_velocity', 'direction']].values
            return torch.FloatTensor(obs)
        else:
            return torch.empty(0, self.config.input_features)
    def process_sequence(self):
        local_predictions = []
        for batch in self.dataloader:
            input_seq, output_seq, object_ids, time_steps = batch
            input_seq = input_seq.to(self.config.device)
            output_seq = output_seq.to(self.config.device)

            # Predict trajectory using the model
            predictions = self.trajectory_model(input_seq)
            local_predictions.append((object_ids, predictions.cpu().numpy(), time_steps))

            with torch.inference_mode():
                prediction = self.trajectory_model(input_seq)
                local_predictions.append((predictions, object_ids, time_steps))

            return local_predictions
        
    def run_camera_thread(self, sync_manager):
        """Main processing loop for camera thread"""
        print(f"Camera {self.camera_id} thread started")
        
        while True:
            try:
                # Process local sequences
                local_preds = self.process_sequences()
                self.local_predictions = local_preds
                
                # Check if it's time to synchronize
                if self.current_time_step % self.config.sync_interval == 0:
                    # Extract observations at current time step
                    observations = self.extract_time_step_observations(self.current_time_step)
                    
                    # Send to synchronization manager
                    self.outgoing_queue.put({
                        'camera_id': self.camera_id,
                        'time_step': self.current_time_step,
                        'observations': observations,
                        'local_predictions': local_preds
                    })
                    
                    # Wait for global context
                    try:
                        global_context = self.incoming_queue.get(timeout=10.0)
                        print(f"Camera {self.camera_id} received global context at time {self.current_time_step}")
                    except queue.Empty:
                        print(f"Camera {self.camera_id} timeout waiting for global context")
                
                self.current_time_step += 1
                time.sleep(0.1)  # Simulate processing time
                
            except Exception as e:
                print(f"Error in camera {self.camera_id}: {e}")
                break

    
class SynchronizationManager:
    """Manages synchronization and global model coordination"""
    def __init__(self, config: PipelineConfig, total_objects: int):
        self.config = config
        self.total_objects = total_objects
        # Global models
        self.compression_model = CompressionModel(
            input_dim=config.input_features,
            compressed_dim=128  # TODO: Make configurable
        )
        
        self.integration_model = GlobalIntegrationModel(
            local_pred_dim=config.position_features,
            compressed_dim=128,
            output_dim=config.position_features,
            total_objects=total_objects
        )
        
        # Camera processors
        self.camera_processors = {}
        self.camera_queues = {}

    def add_camera(self, camera_id: int, df: pd.DataFrame):
        """Add a camera processor"""
        processor = CameraProcessor(camera_id, df, self.config)
        self.camera_processors[camera_id] = processor
        self.camera_queues[camera_id] = processor.outgoing_queue
        return processor
    
    def synchronization_loop(self):
        """Main synchronization loop"""
        print("Synchronization manager started")
        current_sync_time = 0

        while True:
            try:
                camera_data = {}
                camera_observations = {}

                cameras_reported = set()

                cameras_reported = set()
                while len(cameras_reported) < self.config.num_cameras:
                    for camera_id, camera_queue in self.camera_queues.items():
                        if camera_id not in cameras_reported:
                            try:
                                data = camera_queue.get(timeout=1.0)
                                if data['time_step'] == current_sync_time:
                                    camera_data[camera_id] = data
                                    camera_observations[camera_id] = data['observations']
                                    cameras_reported.add(camera_id)
                            except queue.Empty:
                                continue
                print(f"Synchronization at time step {current_sync_time}")
                
                # Compress global observations using compression model
                global_context = self.compression_model(camera_observations)
                
                # Process each camera's local predictions with global context
                for camera_id, data in camera_data.items():
                    local_preds = data['local_predictions']
                # Integrate local predictions with global context
                    if local_preds:
                        # Extract prediction tensors (simplified)
                        pred_tensors = [pred[0] for pred in local_preds]
                        if pred_tensors:
                            combined_preds = torch.cat(pred_tensors, dim=0)
                            
                            # Generate global predictions for all objects
                            all_object_predictions = self.integration_model(combined_preds, global_context)
                            
                            # Send global context back to camera
                            processor = self.camera_processors[camera_id]
                            processor.incoming_queue.put({
                                'global_context': global_context,
                                'all_predictions': all_object_predictions,
                                'time_step': current_sync_time
                            })
                
                current_sync_time += self.config.sync_interval
            except Exception as e:
                print(f"Error in synchronization manager: {e}")
                break
class TrajectoryPredictionPipeline:
    """Main pipeline orchestrator"""
    def __init__(self, config: PipelineConfig, total_objects: int = 1000):
        self.config = config
        self.total_objects = total_objects
        self.sync_manager = SynchronizationManager(config, total_objects)
        self.camera_threads = []

    def add_camera_data(self, camera_id: int, df: pd.DataFrame):
        """Add data for a specific camera"""
        processor = self.sync_manager.add_camera(camera_id, df)
        return processor
    def start_pipeline(self):
        """Start all camera threads and synchronization"""
        # Start synchronization manager in separate thread
        sync_thread = threading.Thread(target=self.sync_manager.synchronization_loop)
        sync_thread.daemon = True
        sync_thread.start()
        for camera_id, processor in self.sync_manager.camera_processors.items():
            camera_thread = threading.Thread(
                target=processor.run_camera_thread,
                args=(self.sync_manager,)
            )
            camera_thread.daemon = True
            camera_thread.start()
            self.camera_threads.append(camera_thread)
        
        print("Pipeline started successfully")
        return sync_thread, self.camera_threads
    
    def get_models(self):
        """Get access to models for training/inference"""
        return {
            'compression_model': self.sync_manager.compression_model,
            'trajectory_model': list(self.sync_manager.camera_processors.values())[0].trajectory_model,
            'integration_model': self.sync_manager.integration_model
        }


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = PipelineConfig(
        sequence_length=50,
        input_length=30,
        output_length=20,
        sync_interval=5,
        batch_size=32
    )
    
    # Create pipeline
    pipeline = TrajectoryPredictionPipeline(config)
    
    # Example: Load your camera data (replace with actual DataFrames)
    # df1, df2, df3, df4 = load_camera_data()  # Your camera dataframes
    
    # Add camera data (uncomment when you have real data)
    # pipeline.add_camera_data(0, df1)
    # pipeline.add_camera_data(1, df2)
    # pipeline.add_camera_data(2, df3)
    # pipeline.add_camera_data(3, df4)
    
    # Start pipeline
    # sync_thread, camera_threads = pipeline.start_pipeline()
    
    # Get models for training
    # models = pipeline.get_models()
    
    print("Pipeline scaffolding created successfully!")
    print("TODO: Implement the three model architectures:")
    print("1. CompressionModel - compress multi-camera observations")
    print("2. TrajectoryPredictionModel - predict trajectories from sequences")
    print("3. GlobalIntegrationModel - integrate local predictions with global context")