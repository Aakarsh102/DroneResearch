import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import json
import shutil
from typing import List, Tuple, Dict, Any, Optional
import math
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pickle

class DroneSequencePipeline:
    def __init__(self, multi_drone_dataset_path: str, output_path: str = "sequence_data"):
        """
        Initialize the sequence pipeline for multi-drone data
        
        Args:
            multi_drone_dataset_path: Path to the processed multi-drone dataset
            output_path: Path where sequence data will be saved
        """
        self.dataset_path = Path(multi_drone_dataset_path)
        self.output_path = Path(output_path)
        self.fps = 2.5  # Stanford drone dataset FPS
        self.sequence_length = 8  # seconds
        self.frames_per_sequence = int(self.fps * self.sequence_length)  # 20 frames
        
        # Stanford drone dataset labels
        self.label_map = {
            'Biker': 0,
            'Pedestrian': 1,
            'Skater': 2,
            'Cart': 3,
            'Car': 4,
            'Bus': 5
        }
        
        # Heatmap parameters
        self.heatmap_size = (64, 64)  # Standard size for model training
        self.sigma = 2.0  # Gaussian sigma for heatmap generation
        
    def _load_drone_annotations(self, location: str, drone_id: str) -> Dict[str, pd.DataFrame]:
        """
        Load all annotation files for a specific drone
        
        Args:
            location: Location name
            drone_id: Drone identifier
            
        Returns:
            Dictionary mapping video names to annotation DataFrames
        """
        annotations = {}
        drone_path = self.dataset_path / location / drone_id
        
        if not drone_path.exists():
            return annotations
        
        for file in drone_path.glob("*_annotations.txt"):
            video_name = file.stem.replace("_annotations", "")
            try:
                df = pd.read_csv(file, sep='\s+', header=None,
                               names=['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 
                                     'frame', 'lost', 'occluded', 'generated', 'label'])
                annotations[video_name] = df
            except Exception as e:
                print(f"Error loading annotations for {video_name}: {e}")
        
        return annotations
    
    def _load_drone_metadata(self, location: str, drone_id: str) -> Dict[str, Any]:
        """
        Load metadata for all videos of a drone
        
        Args:
            location: Location name
            drone_id: Drone identifier
            
        Returns:
            Dictionary mapping video names to metadata
        """
        metadata = {}
        drone_path = self.dataset_path / location / drone_id
        
        for file in drone_path.glob("*_metadata.json"):
            video_name = file.stem.replace("_metadata", "")
            try:
                with open(file, 'r') as f:
                    metadata[video_name] = json.load(f)
            except Exception as e:
                print(f"Error loading metadata for {video_name}: {e}")
        
        return metadata
    
    def _create_heatmap(self, annotations: pd.DataFrame, frame_num: int, 
                       image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a heatmap for a specific frame
        
        Args:
            annotations: Annotations DataFrame
            frame_num: Frame number
            image_shape: (height, width) of the original image
            
        Returns:
            Heatmap as numpy array
        """
        # Filter annotations for this frame
        frame_annotations = annotations[annotations['frame'] == frame_num]
        
        if frame_annotations.empty:
            return np.zeros(self.heatmap_size, dtype=np.float32)
        
        # Create heatmap
        heatmap = np.zeros(image_shape, dtype=np.float32)
        
        for _, row in frame_annotations.iterrows():
            # Skip if object is lost or occluded
            if row['lost'] == 1 or row['occluded'] == 1:
                continue
            
            # Calculate center point
            center_x = (row['xmin'] + row['xmax']) / 2
            center_y = (row['ymin'] + row['ymax']) / 2
            
            # Add Gaussian blob at center
            x_coords = np.arange(image_shape[1])
            y_coords = np.arange(image_shape[0])
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Create Gaussian
            gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * self.sigma**2))
            
            # Add to heatmap with label-based intensity
            label = row['label']
            intensity = self.label_map.get(label, 0) + 1  # +1 to avoid zero intensity
            heatmap += gaussian * intensity
        
        # Resize to standard heatmap size
        heatmap_resized = cv2.resize(heatmap, self.heatmap_size)
        
        # Normalize
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()
        
        return heatmap_resized.astype(np.float32)
    
    def _create_multi_class_heatmap(self, annotations: pd.DataFrame, frame_num: int, 
                                  image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create multi-class heatmaps for a specific frame
        
        Args:
            annotations: Annotations DataFrame
            frame_num: Frame number
            image_shape: (height, width) of the original image
            
        Returns:
            Multi-class heatmap as numpy array (H, W, num_classes)
        """
        # Filter annotations for this frame
        frame_annotations = annotations[annotations['frame'] == frame_num]
        
        num_classes = len(self.label_map)
        heatmaps = np.zeros((image_shape[0], image_shape[1], num_classes), dtype=np.float32)
        
        if frame_annotations.empty:
            return cv2.resize(heatmaps, self.heatmap_size + (num_classes,))
        
        for _, row in frame_annotations.iterrows():
            # Skip if object is lost or occluded
            if row['lost'] == 1 or row['occluded'] == 1:
                continue
            
            # Get class index
            label = row['label']
            if label not in self.label_map:
                continue
            
            class_idx = self.label_map[label]
            
            # Calculate center point
            center_x = (row['xmin'] + row['xmax']) / 2
            center_y = (row['ymin'] + row['ymax']) / 2
            
            # Add Gaussian blob at center
            x_coords = np.arange(image_shape[1])
            y_coords = np.arange(image_shape[0])
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Create Gaussian
            gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * self.sigma**2))
            
            # Add to appropriate class channel
            heatmaps[:, :, class_idx] += gaussian
        
        # Resize each class channel
        resized_heatmaps = np.zeros(self.heatmap_size + (num_classes,), dtype=np.float32)
        for c in range(num_classes):
            resized_heatmaps[:, :, c] = cv2.resize(heatmaps[:, :, c], self.heatmap_size)
            
            # Normalize each class channel
            if resized_heatmaps[:, :, c].max() > 0:
                resized_heatmaps[:, :, c] = resized_heatmaps[:, :, c] / resized_heatmaps[:, :, c].max()
        
        return resized_heatmaps
    
    def _create_sequence(self, annotations: pd.DataFrame, start_frame: int, 
                        image_shape: Tuple[int, int], multi_class: bool = False) -> np.ndarray:
        """
        Create a sequence of heatmaps
        
        Args:
            annotations: Annotations DataFrame
            start_frame: Starting frame number
            image_shape: (height, width) of the original image
            multi_class: Whether to create multi-class heatmaps
            
        Returns:
            Sequence of heatmaps (T, H, W) or (T, H, W, C) for multi-class
        """
        if multi_class:
            num_classes = len(self.label_map)
            sequence = np.zeros((self.frames_per_sequence, self.heatmap_size[0], 
                               self.heatmap_size[1], num_classes), dtype=np.float32)
        else:
            sequence = np.zeros((self.frames_per_sequence, self.heatmap_size[0], 
                               self.heatmap_size[1]), dtype=np.float32)
        
        for i in range(self.frames_per_sequence):
            frame_num = start_frame + i
            
            if multi_class:
                heatmap = self._create_multi_class_heatmap(annotations, frame_num, image_shape)
            else:
                heatmap = self._create_heatmap(annotations, frame_num, image_shape)
            
            sequence[i] = heatmap
        
        return sequence
    
    def _get_frame_range(self, annotations: pd.DataFrame) -> Tuple[int, int]:
        """
        Get the frame range for annotations
        
        Args:
            annotations: Annotations DataFrame
            
        Returns:
            (min_frame, max_frame) tuple
        """
        if annotations.empty:
            return 0, 0
        
        return int(annotations['frame'].min()), int(annotations['frame'].max())
    
    def _save_sequence(self, sequence: np.ndarray, location: str, drone_id: str, 
                      video_name: str, sequence_id: int, start_frame: int, 
                      sequence_type: str = "single_class"):
        """
        Save sequence data
        
        Args:
            sequence: Sequence array
            location: Location name
            drone_id: Drone identifier
            video_name: Video name
            sequence_id: Sequence identifier
            start_frame: Starting frame number
            sequence_type: Type of sequence (single_class or multi_class)
        """
        # Create directory structure
        output_dir = self.output_path / sequence_type / location / drone_id / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sequence
        sequence_file = output_dir / f"sequence_{sequence_id:04d}.npy"
        np.save(sequence_file, sequence)
        
        # Save metadata
        metadata = {
            'location': location,
            'drone_id': drone_id,
            'video_name': video_name,
            'sequence_id': sequence_id,
            'start_frame': start_frame,
            'end_frame': start_frame + self.frames_per_sequence - 1,
            'sequence_length': self.frames_per_sequence,
            'fps': self.fps,
            'duration_seconds': self.sequence_length,
            'heatmap_size': self.heatmap_size,
            'sequence_type': sequence_type,
            'label_map': self.label_map
        }
        
        metadata_file = output_dir / f"sequence_{sequence_id:04d}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_all_drones(self, multi_class: bool = False, overlap_ratio: float = 0.5):
        """
        Process all drones and create sequences
        
        Args:
            multi_class: Whether to create multi-class heatmaps
            overlap_ratio: Overlap ratio between consecutive sequences (0-1)
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        sequence_type = "multi_class" if multi_class else "single_class"
        
        # Calculate step size for overlapping sequences
        step_size = int(self.frames_per_sequence * (1 - overlap_ratio))
        
        total_sequences = 0
        
        # Process each location
        for location_dir in self.dataset_path.iterdir():
            if not location_dir.is_dir():
                continue
            
            location = location_dir.name
            print(f"Processing location: {location}")
            
            # Process each drone
            for drone_dir in location_dir.iterdir():
                if not drone_dir.is_dir():
                    continue
                
                drone_id = drone_dir.name
                print(f"  Processing {drone_id}")
                
                # Load annotations and metadata
                annotations = self._load_drone_annotations(location, drone_id)
                metadata = self._load_drone_metadata(location, drone_id)
                
                # Process each video
                for video_name, video_annotations in annotations.items():
                    if video_annotations.empty:
                        continue
                    
                    # Get image shape from metadata
                    video_metadata = metadata.get(video_name, {})
                    fov_shape = video_metadata.get('fov_image_shape')
                    
                    if fov_shape is None:
                        print(f"    Warning: No FOV shape found for {video_name}, skipping")
                        continue
                    
                    image_shape = (fov_shape[0], fov_shape[1])
                    
                    # Get frame range
                    min_frame, max_frame = self._get_frame_range(video_annotations)
                    
                    # Create sequences
                    sequence_id = 0
                    start_frame = min_frame
                    
                    while start_frame + self.frames_per_sequence - 1 <= max_frame:
                        # Create sequence
                        sequence = self._create_sequence(
                            video_annotations, start_frame, image_shape, multi_class
                        )
                        
                        # Save sequence
                        self._save_sequence(
                            sequence, location, drone_id, video_name, 
                            sequence_id, start_frame, sequence_type
                        )
                        
                        sequence_id += 1
                        total_sequences += 1
                        start_frame += step_size
                    
                    print(f"    Created {sequence_id} sequences for {video_name}")
        
        print(f"\nTotal sequences created: {total_sequences}")
        print(f"Output saved to: {self.output_path}")
        
        # Save dataset summary
        self._save_dataset_summary(sequence_type, total_sequences)
    
    def _save_dataset_summary(self, sequence_type: str, total_sequences: int):
        """
        Save dataset summary
        
        Args:
            sequence_type: Type of sequences created
            total_sequences: Total number of sequences
        """
        summary = {
            'sequence_type': sequence_type,
            'total_sequences': total_sequences,
            'sequence_length_frames': self.frames_per_sequence,
            'sequence_length_seconds': self.sequence_length,
            'fps': self.fps,
            'heatmap_size': self.heatmap_size,
            'label_map': self.label_map,
            'num_classes': len(self.label_map)
        }
        
        summary_file = self.output_path / f"dataset_summary_{sequence_type}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def visualize_sequence(self, sequence_path: str, save_path: str = None):
        """
        Visualize a sequence of heatmaps
        
        Args:
            sequence_path: Path to the sequence .npy file
            save_path: Path to save visualization (optional)
        """
        sequence = np.load(sequence_path)
        
        # Handle both single-class and multi-class sequences
        if len(sequence.shape) == 3:  # Single-class (T, H, W)
            fig, axes = plt.subplots(4, 5, figsize=(15, 12))
            axes = axes.flatten()
            
            for i in range(min(20, sequence.shape[0])):
                axes[i].imshow(sequence[i], cmap='hot', interpolation='nearest')
                axes[i].set_title(f'Frame {i+1}')
                axes[i].axis('off')
            
            plt.tight_layout()
            
        else:  # Multi-class (T, H, W, C)
            num_classes = sequence.shape[3]
            fig, axes = plt.subplots(num_classes, 5, figsize=(15, 3*num_classes))
            
            # Show first 5 frames for each class
            for class_idx in range(num_classes):
                class_name = [k for k, v in self.label_map.items() if v == class_idx][0]
                
                for frame_idx in range(min(5, sequence.shape[0])):
                    if num_classes == 1:
                        ax = axes[frame_idx]
                    else:
                        ax = axes[class_idx, frame_idx]
                    
                    ax.imshow(sequence[frame_idx, :, :, class_idx], cmap='hot', interpolation='nearest')
                    if frame_idx == 0:
                        ax.set_ylabel(class_name)
                    ax.set_title(f'Frame {frame_idx+1}')
                    ax.axis('off')
            
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

def main():
    """
    Example usage of the DroneSequencePipeline
    """
    # Initialize pipeline
    pipeline = DroneSequencePipeline("new_stanford_data")
    
    # Process all drones and create single-class sequences
    print("Creating single-class sequences...")
    pipeline.process_all_drones(multi_class=False, overlap_ratio=0.5)
    
    # Process all drones and create multi-class sequences
    print("\nCreating multi-class sequences...")
    pipeline.process_all_drones(multi_class=True, overlap_ratio=0.5)
    
    # Example visualization (uncomment to use)
    # pipeline.visualize_sequence("sequence_data/single_class/bookstore/drone1/video0/sequence_0000.npy")

if __name__ == "__main__":
    main()