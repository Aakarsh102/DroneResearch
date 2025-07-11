import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import json
import shutil
from typing import List, Tuple, Dict, Any
import math
from pathlib import Path

class DroneSimulator:
    def __init__(self, dataset_path: str, output_path: str = "square_stanford_data_tracks"):
        """
        Initialize the drone simulator
        
        Args:
            dataset_path: Path to the Stanford drone dataset
            output_path: Path where the new multi-drone dataset will be created
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.locations = self._get_locations()
        self.default_fov_side = 420  # Default FOV side length in pixels (2 * previous radius)
        
        # Label to number mapping
        self.label_mapping = {
            'Pedestrian': 1,
            'Biker': 2,
            'Skater': 3,
            'Cart': 4,
            'Car': 5,
            'Bus': 6,
            'Person': 1,  # Same as Pedestrian
            'Bike': 2,    # Same as Biker
            'Default': 9
        }
        
    def _get_locations(self) -> List[str]:
        """Get all available locations from the dataset"""
        annotations_path = self.dataset_path / "annotations"
        locations = []
        
        if annotations_path.exists():
            for item in annotations_path.iterdir():
                if item.is_dir():
                    locations.append(item.name)
        
        return locations
    
    def _check_fov_overlap(self, drone1_pos: Tuple[float, float], drone2_pos: Tuple[float, float], 
                          fov_side: float) -> bool:
        """
        Check if two drones' square FOVs overlap
        
        Args:
            drone1_pos: (x, y) position of drone 1
            drone2_pos: (x, y) position of drone 2
            fov_side: FOV side length for both drones
            
        Returns:
            True if FOVs overlap, False otherwise
        """
        half_side = fov_side / 2
        
        # Calculate bounds for drone 1
        drone1_left = drone1_pos[0] - half_side
        drone1_right = drone1_pos[0] + half_side
        drone1_top = drone1_pos[1] - half_side
        drone1_bottom = drone1_pos[1] + half_side
        
        # Calculate bounds for drone 2
        drone2_left = drone2_pos[0] - half_side
        drone2_right = drone2_pos[0] + half_side
        drone2_top = drone2_pos[1] - half_side
        drone2_bottom = drone2_pos[1] + half_side
        
        # Check for overlap
        x_overlap = drone1_right > drone2_left and drone1_left < drone2_right
        y_overlap = drone1_bottom > drone2_top and drone1_top < drone2_bottom
        
        return x_overlap and y_overlap
    
    def _validate_drone_positions(self, positions: List[Tuple[float, float]], 
                                fov_side: float) -> bool:
        """
        Validate that no drone FOVs overlap
        
        Args:
            positions: List of (x, y) positions for drones
            fov_side: FOV side length for all drones
            
        Returns:
            True if no overlaps, False otherwise
        """
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if self._check_fov_overlap(positions[i], positions[j], fov_side):
                    return False
        return True
    
    def _load_annotations(self, location: str) -> Dict[str, pd.DataFrame]:
        """
        Load all annotation files for a location
        
        Args:
            location: Location name (e.g., 'bookstore', 'coupa')
            
        Returns:
            Dictionary mapping video names to annotation DataFrames
        """
        annotations = {}
        location_path = self.dataset_path / "annotations" / location
        
        if not location_path.exists():
            print(f"Location {location} not found in dataset")
            return annotations
        
        for video_dir in location_path.iterdir():
            if video_dir.is_dir():
                annotation_file = video_dir / "annotations.txt"
                if annotation_file.exists():
                    try:
                        # Load annotations - assuming space-separated format
                        # Common format: trackId xmin ymin xmax ymax frame lost occluded generated label
                        df = pd.read_csv(annotation_file, sep='\s+', header=None,
                                       names=['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 
                                             'frame', 'lost', 'occluded', 'generated', 'label'])
                        annotations[video_dir.name] = df
                    except Exception as e:
                        print(f"Error loading annotations for {video_dir.name}: {e}")
        
        return annotations
    
    def _load_reference_image(self, location: str, video: str) -> np.ndarray:
        """
        Load reference image for a location/video
        
        Args:
            location: Location name
            video: Video name
            
        Returns:
            Reference image as numpy array
        """
        ref_path = self.dataset_path / "annotations" / location / video / "reference.jpg"
        if ref_path.exists():
            return cv2.imread(str(ref_path))
        return None
    
    def _transform_to_local_coordinates(self, annotations: pd.DataFrame, 
                                      drone_pos: Tuple[float, float],
                                      fov_side: float) -> pd.DataFrame:
        """
        Transform annotations to local drone coordinates and filter by square FOV
        
        Args:
            annotations: Original annotations DataFrame
            drone_pos: (x, y) position of the drone
            fov_side: FOV side length
            
        Returns:
            Filtered and transformed annotations DataFrame
        """
        if annotations.empty:
            return annotations
        
        # Calculate center points of bounding boxes
        annotations = annotations.copy()
        annotations['center_x'] = (annotations['xmin'] + annotations['xmax']) / 2
        annotations['center_y'] = (annotations['ymin'] + annotations['ymax']) / 2
        
        # Filter objects within square FOV
        half_side = fov_side / 2
        x_in_fov = (annotations['center_x'] >= drone_pos[0] - half_side) & \
                   (annotations['center_x'] <= drone_pos[0] + half_side)
        y_in_fov = (annotations['center_y'] >= drone_pos[1] - half_side) & \
                   (annotations['center_y'] <= drone_pos[1] + half_side)
        
        in_fov = x_in_fov & y_in_fov
        filtered_annotations = annotations[in_fov].copy()
        
        if filtered_annotations.empty:
            return filtered_annotations
        
        # Transform to local coordinates (drone at origin)
        filtered_annotations['local_xmin'] = filtered_annotations['xmin'] - drone_pos[0]
        filtered_annotations['local_ymin'] = filtered_annotations['ymin'] - drone_pos[1]
        filtered_annotations['local_xmax'] = filtered_annotations['xmax'] - drone_pos[0]
        filtered_annotations['local_ymax'] = filtered_annotations['ymax'] - drone_pos[1]
        filtered_annotations['local_center_x'] = filtered_annotations['center_x'] - drone_pos[0]
        filtered_annotations['local_center_y'] = filtered_annotations['center_y'] - drone_pos[1]
        
        return filtered_annotations
    
    def _create_fov_image(self, reference_image: np.ndarray, 
                         drone_pos: Tuple[float, float],
                         fov_side: float) -> np.ndarray:
        """
        Create cropped square FOV image for a drone
        
        Args:
            reference_image: Original reference image
            drone_pos: (x, y) position of the drone
            fov_side: FOV side length
            
        Returns:
            Cropped square FOV image
        """
        if reference_image is None:
            return None
        
        height, width = reference_image.shape[:2]
        half_side = fov_side / 2
        
        # Calculate square bounds
        x_min = max(0, int(drone_pos[0] - half_side))
        x_max = min(width, int(drone_pos[0] + half_side))
        y_min = max(0, int(drone_pos[1] - half_side))
        y_max = min(height, int(drone_pos[1] + half_side))
        
        # Create square mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(reference_image, reference_image, mask=mask)
        
        # Crop to square FOV bounds
        cropped_fov = masked_image[y_min:y_max, x_min:x_max]
        
        return cropped_fov
    
    def _generate_heatmap_sequences(self, annotations: pd.DataFrame, 
                                   drone_pos: Tuple[float, float],
                                   fov_side: float) -> List[np.ndarray]:
        """
        Generate 8-second heatmap sequences with 4-second overlaps
        
        Args:
            annotations: Annotations DataFrame with local coordinates
            drone_pos: (x, y) position of the drone
            fov_side: FOV side length
            
        Returns:
            List of 2D arrays representing heatmap sequences
        """
        if annotations.empty:
            return []
        
        # Filter annotations to only include those in FOV
        local_annotations = self._transform_to_local_coordinates(annotations, drone_pos, fov_side)
        
        if local_annotations.empty:
            return []
        
        # Get frame range
        min_frame = local_annotations['frame'].min()
        max_frame = local_annotations['frame'].max()
        
        # Create sequences: 8 seconds = 20 frames, overlap = 4 seconds = 10 frames
        sequence_length = 20  # 8 seconds at 2.5 Hz
        overlap = 10  # 4 seconds at 2.5 Hz
        step = sequence_length - overlap  # 10 frames
        
        sequences = []
        
        for start_frame in range(min_frame, max_frame - sequence_length + 1, step):
            end_frame = start_frame + sequence_length
            
            # Get annotations for this sequence
            sequence_annotations = local_annotations[
                (local_annotations['frame'] >= start_frame) & 
                (local_annotations['frame'] < end_frame)
            ]
            
            if sequence_annotations.empty:
                continue
            
            # Create heatmap for this sequence
            heatmap = self._create_sequence_heatmap(sequence_annotations, fov_side)
            
            if heatmap is not None:
                sequences.append(heatmap)
        
        return sequences
    
    def _create_sequence_heatmap(self, sequence_annotations: pd.DataFrame, 
                                fov_side: float) -> np.ndarray:
        """
        Create a heatmap for a single sequence showing object tracks
        
        Args:
            sequence_annotations: Annotations for the sequence
            fov_side: FOV side length
            
        Returns:
            2D array with [id, label] values at object center positions
        """
        if sequence_annotations.empty:
            return None
        
        # Create heatmap canvas (square FOV)
        heatmap_size = int(fov_side)
        heatmap = np.zeros((heatmap_size, heatmap_size, 2), dtype=np.int32)  # [id, label]
        
        # Transform local coordinates to heatmap coordinates
        half_side = fov_side / 2
        
        for _, row in sequence_annotations.iterrows():
            # Convert local coordinates to heatmap coordinates
            x = int(row['local_center_x'] + half_side)
            y = int(row['local_center_y'] + half_side)
            
            # Ensure coordinates are within bounds
            if 0 <= x < heatmap_size and 0 <= y < heatmap_size:
                track_id = int(row['trackId'])
                label = row['label'].strip('"')  # Remove quotes
                label_num = self.label_mapping.get(label, self.label_mapping['Default'])
                
                # Set [id, label] at this position
                heatmap[y, x, 0] = track_id
                heatmap[y, x, 1] = label_num
        
        return heatmap
    
    def _save_heatmap_sequences(self, sequences: List[np.ndarray], 
                               location: str, drone_id: str, video: str):
        """
        Save heatmap sequences to files
        
        Args:
            sequences: List of heatmap arrays
            location: Location name
            drone_id: Drone identifier
            video: Video name
        """
        if not sequences:
            return
        
        # Create heatmap directory
        heatmap_dir = self.output_path / location / drone_id / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each sequence
        for seq_idx, heatmap in enumerate(sequences):
            # Save as .npy file for easy loading
            heatmap_file = heatmap_dir / f"{video}_sequence_{seq_idx:03d}.npy"
            np.save(heatmap_file, heatmap)
            
            # Also save metadata about the sequence
            metadata = {
                'sequence_index': seq_idx,
                'video': video,
                'drone_id': drone_id,
                'location': location,
                'shape': heatmap.shape,
                'data_format': '[id, label]',
                'label_mapping': self.label_mapping
            }
            
            metadata_file = heatmap_dir / f"{video}_sequence_{seq_idx:03d}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _save_drone_data(self, location: str, drone_id: str, video: str,
                        local_annotations: pd.DataFrame, fov_image: np.ndarray):
        """
        Save drone-specific data to the output directory
        
        Args:
            location: Location name
            drone_id: Drone identifier
            video: Video name
            local_annotations: Transformed annotations
            fov_image: FOV image
        """
        # Create directory structure
        drone_dir = self.output_path / location / drone_id
        drone_dir.mkdir(parents=True, exist_ok=True)
        
        # Save annotations
        if not local_annotations.empty:
            annotation_file = drone_dir / f"{video}_annotations.txt"
            # Save with local coordinates
            columns_to_save = ['trackId', 'local_xmin', 'local_ymin', 'local_xmax', 'local_ymax',
                             'frame', 'lost', 'occluded', 'generated', 'label']
            local_annotations[columns_to_save].to_csv(annotation_file, sep=' ', 
                                                    header=False, index=False)
        
        # Save FOV image
        if fov_image is not None and fov_image.size != 0:
            fov_file = drone_dir / f"{video}_fov.jpg"
            cv2.imwrite(str(fov_file), fov_image)
        
        # Save metadata
        metadata = {
            'location': location,
            'drone_id': drone_id,
            'video': video,
            'total_annotations': len(local_annotations),
            'fov_image_shape': fov_image.shape if fov_image is not None else None,
            'fov_type': 'square',
            'label_mapping': self.label_mapping
        }
        
        metadata_file = drone_dir / f"{video}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def simulate_drones(self, drone_positions: Dict[str, List[Tuple[float, float]]], 
                       fov_side: float = None):
        """
        Simulate multiple drones for each location with square FOVs and generate heatmaps
        
        Args:
            drone_positions: Dictionary mapping location names to lists of (x, y) positions
            fov_side: FOV side length for all drones (uses default if None)
        """
        if fov_side is None:
            fov_side = self.default_fov_side
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        for location, positions in drone_positions.items():
            if location not in self.locations:
                print(f"Warning: Location '{location}' not found in dataset")
                continue
            
            # Validate drone positions
            if not self._validate_drone_positions(positions, fov_side):
                print(f"Warning: Some drone square FOVs overlap in location '{location}'")
                continue
            
            print(f"Processing location: {location}")
            
            # Load annotations for this location
            annotations = self._load_annotations(location)
            
            if not annotations:
                print(f"No annotations found for location: {location}")
                continue
            
            # Process each drone
            for drone_idx, drone_pos in enumerate(positions):
                drone_id = f"drone{drone_idx + 1}"
                print(f"  Processing {drone_id} at position {drone_pos}")
                
                # Process each video for this drone
                for video_name, video_annotations in annotations.items():
                    # Load reference image
                    reference_image = self._load_reference_image(location, video_name)
                    
                    # Transform annotations to local coordinates
                    local_annotations = self._transform_to_local_coordinates(
                        video_annotations, drone_pos, fov_side)
                    
                    # Create square FOV image
                    fov_image = self._create_fov_image(reference_image, drone_pos, fov_side)
                    
                    # Generate heatmap sequences
                    heatmap_sequences = self._generate_heatmap_sequences(
                        video_annotations, drone_pos, fov_side)
                    
                    # Save drone data
                    self._save_drone_data(location, drone_id, video_name, 
                                        local_annotations, fov_image)
                    
                    # Save heatmap sequences
                    self._save_heatmap_sequences(heatmap_sequences, location, drone_id, video_name)
                    
                    print(f"    Generated {len(heatmap_sequences)} heatmap sequences for {video_name}")
        
        print("Drone simulation completed!")
        print(f"Output saved to: {self.output_path}")
        print(f"Label mapping: {self.label_mapping}")

def main():
    """
    Example usage of the DroneSimulator with square FOVs and heatmap generation
    """
    # Initialize simulator
    simulator = DroneSimulator("stanford_data/archive")
    
    # Define drone positions for each location
    # Positions are spaced further apart to accommodate square FOVs
    drone_positions = {
        # Bookstore location
        "bookstore": [
        (250, 250),
        (1000, 300),
        (1000, 750),
        (250, 800)
        ],
        
        # Coupa location
        "coupa": [
        (250, 250),
        (900, 300),
        (950, 750),
        (300, 800),
        (1500, 300),
        (1500, 750)
        ],
        
        # Death Circle location
        "deathCircle": [
        (650, 350),
        (1050, 1000),
        (650, 1500),
        (350, 1000)
        ],
        
        # Gates location
        "gates": [
        (650, 350),
        (1050, 1000),
        (650, 1500),
        (350, 1025)
        ],
        
        # Hyang location
        "hyang": [
        (700, 350),
        (1050, 1025),
        (650, 1500),
        (350, 1025)
        ],
        
        # Little location
        "little": [
        (700, 350),
        (1050, 1025),
        (650, 1500),
        (350, 1025)
        ],
        
        # Nexus location
        "nexus": [
        (550, 500),
        (300, 950),   # drone2
        (800, 950),   # drone3
        (550, 1400)
        ],
        
        # Quad location
        "quad": [
        (500, 250),
        (300, 800),   # drone2
        (1150, 750),   # drone3
        (1500, 300)
        ],
    }
    
    # Set FOV side length (adjust based on your needs)
    fov_side = 420  # Square side length in pixels
    
    # Run simulation
    simulator.simulate_drones(drone_positions, fov_side)

if __name__ == "__main__":
    main()