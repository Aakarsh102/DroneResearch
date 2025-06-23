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

@dataclass
class PipelineConfig:
    sequence_length: int = 50
    input_length: int = 30     # t - historical sequence length
    output_length: int = 20    # Not used in new design
    sync_interval: int = 5     # t - synchronization every t time steps
    batch_size: int = 32
    num_cameras: int = 4
    num_devices: int = 4
    position_features: int = 2  # x, y for input and output
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class TrajectoryPredictionModel(nn.Module):
    """Model that takes only x,y positions and predicts next t positions"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, output_steps: int = 10):
        super().__init__()
        self.output_steps = output_steps
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_steps * input_dim)  # Predict t steps of (x,y)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 2) - only x,y positions
        lstm_out, _ = self.lstm(x)
        # Use last timestep output for prediction
        predictions = self.fc(lstm_out[:, -1, :])
        # Reshape to (batch_size, output_steps, 2)
        return predictions.view(-1, self.output_steps, 2)

class ObjectHistoryManager:
    """Manages historical position data for all objects"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        # object_id -> list of (time_step, x, y)
        self.object_positions = defaultdict(list)
        self.max_history_length = config.sync_interval * 10  # Keep more history
        
    def add_position(self, time_step: int, object_id: int, x: float, y: float):
        """Add position observation for an object at a specific time step"""
        self.object_positions[object_id].append((time_step, x, y))
        
        # Keep only recent history, sorted by time
        self.object_positions[object_id].sort(key=lambda x: x[0])
        if len(self.object_positions[object_id]) > self.max_history_length:
            self.object_positions[object_id] = self.object_positions[object_id][-self.max_history_length:]
    
    def get_position_sequence(self, object_id: int, start_time: int, end_time: int) -> Optional[torch.Tensor]:
        """Get position sequence for object from start_time to end_time (inclusive)"""
        if object_id not in self.object_positions:
            return None
            
        positions = self.object_positions[object_id]
        sequence_data = []
        
        for time_step, x, y in positions:
            if start_time <= time_step <= end_time:
                sequence_data.append([x, y])
        
        if len(sequence_data) == 0:
            return None
            
        return torch.FloatTensor(sequence_data)
    
    def get_latest_position(self, object_id: int, time_step: int) -> Optional[Tuple[float, float]]:
        """Get the latest known position for an object at or before time_step"""
        if object_id not in self.object_positions:
            return None
            
        positions = self.object_positions[object_id]
        latest_pos = None
        
        for t, x, y in positions:
            if t <= time_step:
                latest_pos = (x, y)
            else:
                break
                
        return latest_pos

class CameraProcessor:
    def __init__(self, camera_id: int, df: pd.DataFrame, config: PipelineConfig):
        self.camera_id = camera_id
        self.config = config
        # Only keep necessary columns: time_step, object_id, x, y
        self.df = df[['time_step', 'object_id', 'x', 'y']].sort_values(['time_step', 'object_id'])
        
        self.trajectory_model = TrajectoryPredictionModel(
            input_dim=2,  # Only x, y
            output_steps=config.sync_interval
        ).to(config.device)
        
        # Communication queues
        self.outgoing_queue = queue.Queue()  # Send predictions to sync manager
        self.incoming_queue = queue.Queue()  # Receive other cameras' predictions
        
        self.current_time_step = 0
        self.object_history = ObjectHistoryManager(config)
        
        # Initialize history with all available data up to current time
        self._initialize_history()
        
    def _initialize_history(self):
        """Initialize object history with available data"""
        for _, row in self.df.iterrows():
            self.object_history.add_position(
                int(row['time_step']), 
                int(row['object_id']), 
                float(row['x']), 
                float(row['y'])
            )
    
    def get_objects_in_fov(self, time_step: int) -> List[int]:
        """Get object IDs that are in this camera's FOV at given time step"""
        mask = self.df['time_step'] == time_step
        time_step_data = self.df[mask]
        return time_step_data['object_id'].tolist()
    
    def get_all_known_objects(self) -> List[int]:
        """Get all object IDs that this camera has ever seen"""
        return list(self.object_history.object_positions.keys())
    
    def predict_object_trajectory(self, object_id: int, current_time: int, 
                                use_shared_data_at_T: Optional[Dict] = None) -> Optional[torch.Tensor]:
        """
        Predict trajectory for a single object for next t time steps
        
        Args:
            object_id: ID of object to predict
            current_time: Current time step T
            use_shared_data_at_T: Shared position data at time T for objects not in FOV
        """
        # For objects in our FOV: use T-t to T-1 + our observation at T
        if object_id in self.get_objects_in_fov(current_time):
            # Get historical sequence from T-t to T-1
            start_time = current_time - self.config.sync_interval
            end_time = current_time - 1
            historical_seq = self.object_history.get_position_sequence(object_id, start_time, end_time)
            
            # Get our current observation at T
            current_obs = self.object_history.get_position_sequence(object_id, current_time, current_time)
            
            if historical_seq is not None and current_obs is not None:
                # Combine historical + current
                full_sequence = torch.cat([historical_seq, current_obs], dim=0)
            elif current_obs is not None:
                # Only current observation available
                full_sequence = current_obs
            else:
                return None
                
        else:
            # For objects NOT in our FOV: use T-t-1 to T-1 + shared data at T
            start_time = current_time - self.config.sync_interval - 1
            end_time = current_time - 1
            historical_seq = self.object_history.get_position_sequence(object_id, start_time, end_time)
            
            # Use shared position data at T
            if use_shared_data_at_T and object_id in use_shared_data_at_T:
                shared_pos = use_shared_data_at_T[object_id]  # (x, y)
                current_obs = torch.FloatTensor([[shared_pos[0], shared_pos[1]]])
                
                if historical_seq is not None:
                    full_sequence = torch.cat([historical_seq, current_obs], dim=0)
                else:
                    full_sequence = current_obs
            else:
                return None
        
        # Ensure we have enough data points
        if len(full_sequence) < self.config.sync_interval:
            # Pad with the first available position
            padding_needed = self.config.sync_interval - len(full_sequence)
            first_pos = full_sequence[0].unsqueeze(0).repeat(padding_needed, 1)
            full_sequence = torch.cat([first_pos, full_sequence], dim=0)
        elif len(full_sequence) > self.config.sync_interval:
            # Take the last t positions
            full_sequence = full_sequence[-self.config.sync_interval:]
        
        # Predict next t time steps
        input_seq = full_sequence.unsqueeze(0).to(self.config.device)  # Add batch dimension
        
        with torch.no_grad():
            predictions = self.trajectory_model(input_seq)  # Shape: (1, t, 2)
            
        return predictions.squeeze(0).cpu()  # Shape: (t, 2)
    
    def predict_for_objects_in_fov(self, time_step: int) -> Dict[int, Dict]:
        """Predict trajectories for objects currently in FOV"""
        objects_in_fov = self.get_objects_in_fov(time_step)
        predictions = {}
        
        for object_id in objects_in_fov:
            trajectory_pred = self.predict_object_trajectory(object_id, time_step)
            if trajectory_pred is not None:
                # Get current position for sharing
                current_pos = self.object_history.get_position_sequence(object_id, time_step, time_step)
                if current_pos is not None:
                    predictions[object_id] = {
                        'object_id': object_id,  # Include object ID for matching
                        'current_position': current_pos.squeeze().tolist(),  # [x, y]
                        'current_time': time_step,  # Time when this position was observed
                        'future_predictions': trajectory_pred.tolist(),  # [[x1,y1], [x2,y2], ...]
                        'prediction_start_time': time_step + 1,
                        'prediction_end_time': time_step + self.config.sync_interval
                    }
        
        return predictions
    
    def predict_for_all_objects(self, current_time: int, shared_data_time: int, 
                              shared_positions_at_T: Dict[int, Tuple[float, float]]) -> Dict[int, torch.Tensor]:
        """
        Predict trajectories for ALL objects with temporal alignment
        
        Args:
            current_time: Current time T+k when making predictions
            shared_data_time: Original time T when shared data was sent
            shared_positions_at_T: Dict of object_id -> (x, y) positions at time T from other cameras
        """
        all_predictions = {}
        all_known_objects = self.get_all_known_objects()
        
        # Add objects from shared data that we might not know about
        for obj_id in shared_positions_at_T.keys():
            if obj_id not in all_known_objects:
                all_known_objects.append(obj_id)
        
        k = current_time - shared_data_time  # Time offset
        print(f"Camera {self.camera_id}: Predicting at T+{k} using data from T={shared_data_time}")
        
        for object_id in all_known_objects:
            # Use temporal alignment for sequence building
            aligned_sequence = self.get_temporal_aligned_sequence(
                object_id, current_time, shared_data_time
            )
            
            if aligned_sequence is not None:
                # Make prediction using the aligned sequence
                input_seq = aligned_sequence.unsqueeze(0).to(self.config.device)
                
                with torch.no_grad():
                    predictions = self.trajectory_model(input_seq)  # Shape: (1, t, 2)
                    all_predictions[object_id] = predictions.squeeze(0).cpu()  # Shape: (t, 2)
        
        return all_predictions
    
    def update_history_with_shared_data(self, shared_data: Dict[int, Dict[int, Dict]], 
                                       shared_time: int, current_time: int):
        """
        Update object history with shared position data from other cameras
        Handles temporal alignment when shared data arrives at different times
        
        Args:
            shared_data: Data shared by other cameras
            shared_time: Time T when the data was originally shared
            current_time: Current time T+k when we're processing this data
        """
        print(f"Camera {self.camera_id}: Processing shared data from time {shared_time} at current time {current_time}")
        
        for camera_id, camera_predictions in shared_data.items():
            if camera_id != self.camera_id:  # Don't process own data
                for object_id, pred_data in camera_predictions.items():
                    # Extract object info with proper ID matching
                    obj_id = pred_data.get('object_id', object_id)  # Use explicit object_id if available
                    pos = pred_data['current_position']  # [x, y]
                    pos_time = pred_data.get('current_time', shared_time)  # Time when position was observed
                    
                    # Add the position from other camera at the correct timestamp
                    self.object_history.add_position(pos_time, obj_id, pos[0], pos[1])
                    
                    print(f"  Added position for object {obj_id} at time {pos_time}: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    def get_temporal_aligned_sequence(self, object_id: int, prediction_time: int, 
                                    shared_data_time: int) -> Optional[torch.Tensor]:
        """
        Get temporally aligned sequence for prediction when shared data arrives at different times
        
        Args:
            object_id: Object to predict for
            prediction_time: Current time T+k when making prediction
            shared_data_time: Original time T when shared data was sent
        
        Returns:
            Sequence of positions for prediction, properly aligned temporally
        """
        k = prediction_time - shared_data_time  # Time offset
        
        if k < 0:
            # Shared data is from the future - shouldn't happen in normal operation
            print(f"Warning: Shared data from future for object {object_id}")
            return None
        
        if k >= self.config.sync_interval:
            # Shared data is too old to be useful
            print(f"Warning: Shared data too old for object {object_id} (k={k})")
            return None
        
        # We need t total time steps for prediction
        # Use (k+1) new data points from T to T+k and (t-k-1) old data points
        new_data_start = shared_data_time
        new_data_end = prediction_time
        old_data_start = shared_data_time - self.config.sync_interval + k
        old_data_end = shared_data_time - 1
        
        # Get old data sequence (T-t+k to T-1)
        old_sequence = None
        if old_data_start <= old_data_end:
            old_sequence = self.object_history.get_position_sequence(
                object_id, old_data_start, old_data_end
            )
        
        # Get new data sequence (T to T+k)
        new_sequence = self.object_history.get_position_sequence(
            object_id, new_data_start, new_data_end
        )
        
        # Combine sequences
        if old_sequence is not None and new_sequence is not None:
            full_sequence = torch.cat([old_sequence, new_sequence], dim=0)
        elif new_sequence is not None:
            full_sequence = new_sequence
        elif old_sequence is not None:
            full_sequence = old_sequence
        else:
            return None
        
        # Ensure we have exactly t time steps
        target_length = self.config.sync_interval
        if len(full_sequence) < target_length:
            # Pad with first position
            padding_needed = target_length - len(full_sequence)
            first_pos = full_sequence[0].unsqueeze(0).repeat(padding_needed, 1)
            full_sequence = torch.cat([first_pos, full_sequence], dim=0)
        elif len(full_sequence) > target_length:
            # Take the most recent t positions
            full_sequence = full_sequence[-target_length:]
        
        print(f"  Object {object_id}: Using {len(old_sequence) if old_sequence is not None else 0} old + {len(new_sequence) if new_sequence is not None else 0} new data points")
        
        return full_sequence
    
    def run_camera_thread(self):
        """Main processing loop for camera thread"""
        print(f"Camera {self.camera_id} thread started")
        
        while True:
            try:
                # Check if it's time to synchronize
                if self.current_time_step % self.config.sync_interval == 0:
                    print(f"Camera {self.camera_id} synchronizing at time {self.current_time_step}")
                    
                    # Step 1: Predict for objects in our FOV from T to T+t
                    # Only share predictions for objects in our FOV
                    our_fov_predictions = self.predict_for_objects_in_fov(self.current_time_step)
                    
                    # Step 2: Send only our FOV objects' predictions to sync manager
                    self.outgoing_queue.put({
                        'camera_id': self.camera_id,
                        'time_step': self.current_time_step,
                        'predictions': our_fov_predictions,
                        'objects_in_fov': list(our_fov_predictions.keys())
                    })
                    
                    # Step 3: Wait for shared predictions from other cameras
                    try:
                        shared_data = self.incoming_queue.get(timeout=10.0)
                        all_camera_predictions = shared_data['all_camera_predictions']
                        
                        print(f"Camera {self.camera_id} received shared predictions from {len(all_camera_predictions)} cameras")
                        
                        # Step 4: Update our history with shared position data
                        shared_time = shared_data['sync_time_step']
                        self.update_history_with_shared_data(
                            all_camera_predictions, shared_time, self.current_time_step
                        )
                        
                        # Step 5: Extract current positions at T from shared data
                        shared_positions_at_T = {}
                        for camera_id, camera_preds in all_camera_predictions.items():
                            if camera_id != self.camera_id:
                                for obj_id, pred_data in camera_preds.items():
                                    # Use explicit object_id for proper matching
                                    actual_obj_id = pred_data.get('object_id', obj_id)
                                    shared_positions_at_T[actual_obj_id] = tuple(pred_data['current_position'])
                        
                        # Step 6: Now predict for ALL objects with temporal alignment
                        all_predictions = self.predict_for_all_objects(
                            self.current_time_step, 
                            shared_time,
                            shared_positions_at_T
                        )
                        
                        print(f"Camera {self.camera_id} generated predictions for {len(all_predictions)} objects for next interval")
                        
                    except queue.Empty:
                        print(f"Camera {self.camera_id} timeout waiting for shared predictions")
                
                self.current_time_step += 1
                time.sleep(0.1)  # Simulate processing time
                
            except Exception as e:
                print(f"Error in camera {self.camera_id}: {e}")
                import traceback
                traceback.print_exc()
                break

class SynchronizationManager:
    """Manages synchronization and sharing of predictions between cameras"""
    def __init__(self, config: PipelineConfig):
        self.config = config
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
                # Collect predictions from all cameras for their FOV objects only
                camera_predictions = {}
                cameras_reported = set()
                
                print(f"Sync manager waiting for cameras at time {current_sync_time}")
                
                # Wait for all cameras to report their FOV predictions
                timeout_counter = 0
                while len(cameras_reported) < len(self.camera_processors) and timeout_counter < 50:
                    for camera_id, camera_queue in self.camera_queues.items():
                        if camera_id not in cameras_reported:
                            try:
                                data = camera_queue.get(timeout=0.1)
                                if data['time_step'] == current_sync_time:
                                    camera_predictions[camera_id] = data['predictions']
                                    cameras_reported.add(camera_id)
                                    print(f"Received predictions from camera {camera_id} for {len(data['predictions'])} objects in FOV")
                            except queue.Empty:
                                pass
                    timeout_counter += 1
                
                if len(cameras_reported) > 0:
                    print(f"Synchronization at time {current_sync_time}: {len(cameras_reported)} cameras reported")
                    
                    # Share all camera predictions with each camera
                    for camera_id, processor in self.camera_processors.items():
                        try:
                            processor.incoming_queue.put({
                                'all_camera_predictions': camera_predictions,
                                'sync_time_step': current_sync_time
                            }, timeout=1.0)
                        except queue.Full:
                            print(f"Warning: Camera {camera_id} incoming queue is full")
                else:
                    print(f"No cameras reported at time {current_sync_time}")
                
                current_sync_time += self.config.sync_interval
                
            except Exception as e:
                print(f"Error in synchronization manager: {e}")
                import traceback
                traceback.print_exc()
                break

class TrajectoryPredictionPipeline:
    """Main pipeline orchestrator"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sync_manager = SynchronizationManager(config)
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
        
        # Start camera processor threads
        for camera_id, processor in self.sync_manager.camera_processors.items():
            camera_thread = threading.Thread(target=processor.run_camera_thread)
            camera_thread.daemon = True
            camera_thread.start()
            self.camera_threads.append(camera_thread)
        
        print("Pipeline started successfully")
        return sync_thread, self.camera_threads
    
    def get_models(self):
        """Get access to models for training/inference"""
        models = {}
        if self.sync_manager.camera_processors:
            first_processor = list(self.sync_manager.camera_processors.values())[0]
            models['trajectory_model'] = first_processor.trajectory_model
        return models

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = PipelineConfig(
        sequence_length=50,
        input_length=30,
        output_length=20,
        sync_interval=5,
        batch_size=32,
        num_cameras=4
    )
    
    # Create pipeline
    pipeline = TrajectoryPredictionPipeline(config)
    
    def create_dummy_camera_data(camera_id: int, num_objects: int = 10, num_timesteps: int = 100):
        """Create dummy circular motion data - simplified to only necessary columns"""
        data = []
        
        # Each camera sees some unique objects and some shared objects
        base_objects = list(range(camera_id * (num_objects // 2), (camera_id + 1) * (num_objects // 2)))
        shared_objects = list(range(num_objects, num_objects + 5))  # 5 shared objects across all cameras
        all_objects = base_objects + shared_objects
        
        for obj_id in all_objects:
            radius = np.random.uniform(5, 20)
            angular_velocity = np.random.uniform(0.1, 0.5)
            center_x = np.random.uniform(-50, 50)
            center_y = np.random.uniform(-50, 50)
            
            for t in range(num_timesteps):
                angle = angular_velocity * t
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                # Only keep essential columns
                data.append({
                    'time_step': t,
                    'object_id': obj_id,
                    'x': x,
                    'y': y
                })
        
        return pd.DataFrame(data)
    
    # Create dummy data for each camera with overlapping objects
    print("Creating dummy data for testing...")
    for camera_id in range(config.num_cameras):
        dummy_df = create_dummy_camera_data(camera_id)
        pipeline.add_camera_data(camera_id, dummy_df)
        print(f"Camera {camera_id} loaded with {len(dummy_df)} observations")
    
    # Start pipeline
    print("Starting pipeline...")
    sync_thread, camera_threads = pipeline.start_pipeline()
    
    # Let it run for a few seconds for demonstration
    time.sleep(15)
    
    print("Pipeline demonstration completed!")
    print("\nKey features implemented:")
    print("1. Data reduced to only: time_step, object_id, x, y")
    print("2. Model takes only x,y positions as input")
    print("3. Each camera shares only FOV objects' predictions")
    print("4. For FOV objects: uses T-t to T-1 + current T observation")
    print("5. For non-FOV objects: uses T-t-1 to T-1 + shared T position") 
    print("6. All cameras predict for ALL objects but share only FOV objects")
    print("7. Proper object ID tracking and position sharing")