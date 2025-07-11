import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Tuple, Dict

class DroneVisualizer:
    def __init__(self, dataset_path: str):
        """
        Initialize drone visualizer
        
        Args:
            dataset_path: Path to the Stanford drone dataset
        """
        self.dataset_path = Path(dataset_path)
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
    
    def visualize_drone_positions(self, location: str, drone_positions: List[Tuple[float, float]], 
                                fov_radius: float, save_path: str = None):
        """
        Visualize drone positions and FOVs on the reference image
        
        Args:
            location: Location name
            drone_positions: List of (x, y) positions for drones
            fov_radius: FOV radius
            save_path: Optional path to save the visualization
        """
        # Load reference image
        ref_path = self.dataset_path / "annotations" / location / "video4" / "reference.jpg"
        if not ref_path.exists():
            # Try other video directories
            location_path = self.dataset_path / "annotations" / location
            for video_dir in location_path.iterdir():
                if video_dir.is_dir():
                    ref_path = video_dir / "reference.jpg"
                    if ref_path.exists():
                        break
        
        if not ref_path.exists():
            print(f"No reference image found for location: {location}")
            return
        
        # Load and copy image
        image = cv2.imread(str(ref_path))
        if image is None:
            print(f"Could not load reference image: {ref_path}")
            return
        
        vis_image = image.copy()
        
        # Draw drone positions and FOVs
        for i, (x, y) in enumerate(drone_positions):
            color = self.colors[i % len(self.colors)]
            
            # Draw FOV circle
            cv2.circle(vis_image, (int(x), int(y)), int(fov_radius), color, 2)
            cv2.rectangle(vis_image, (int(x - fov_radius), int(y - fov_radius)), (int(x + fov_radius), int(y + fov_radius)), color, 2)  # Small square for FOV center

            # Draw drone position
            cv2.circle(vis_image, (int(x), int(y)), 8, color, -1)
            
            # Add drone label
            cv2.putText(vis_image, f'D{i+1}', (int(x+10), int(y-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Drone Positions and FOVs - {location}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def check_overlaps(self, drone_positions: List[Tuple[float, float]], 
                      fov_radius: float) -> List[Tuple[int, int]]:
        """
        Check for overlapping FOVs and return pairs of overlapping drones
        
        Args:
            drone_positions: List of (x, y) positions for drones
            fov_radius: FOV radius
            
        Returns:
            List of tuples containing indices of overlapping drone pairs
        """
        overlaps = []
        
        for i in range(len(drone_positions)):
            for j in range(i + 1, len(drone_positions)):
                pos1 = drone_positions[i]
                pos2 = drone_positions[j]
                
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance < (2 * fov_radius):
                    overlaps.append((i, j))
        
        return overlaps
    
    def generate_non_overlapping_positions(self, image_shape: Tuple[int, int], 
                                         num_drones: int, fov_radius: float,
                                         max_attempts: int = 1000) -> List[Tuple[float, float]]:
        """
        Generate non-overlapping drone positions within image bounds
        
        Args:
            image_shape: (height, width) of the reference image
            num_drones: Number of drones to place
            fov_radius: FOV radius
            max_attempts: Maximum attempts to generate valid positions
            
        Returns:
            List of (x, y) positions for drones
        """
        height, width = image_shape
        positions = []
        
        for attempt in range(max_attempts):
            if len(positions) >= num_drones:
                break
            
            # Generate random position within image bounds (with margin for FOV)
            x = np.random.randint(fov_radius, width - fov_radius)
            y = np.random.randint(fov_radius, height - fov_radius)
            
            # Check if this position overlaps with existing positions
            valid = True
            for existing_pos in positions:
                distance = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if distance < (2 * fov_radius):
                    valid = False
                    break
            
            if valid:
                positions.append((float(x), float(y)))
        
        if len(positions) < num_drones:
            print(f"Warning: Could only generate {len(positions)} non-overlapping positions out of {num_drones} requested")
        
        return positions
    
    def save_drone_config(self, drone_positions: Dict[str, List[Tuple[float, float]]], 
                         fov_radius: float, filename: str = "drone_config.json"):
        """
        Save drone configuration to JSON file
        
        Args:
            drone_positions: Dictionary mapping location names to drone positions
            fov_radius: FOV radius
            filename: Output filename
        """
        config = {
            "fov_radius": fov_radius,
            "drone_positions": drone_positions
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Drone configuration saved to: {filename}")
    
    def load_drone_config(self, filename: str = "drone_config.json") -> Dict:
        """
        Load drone configuration from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            Dictionary containing drone configuration
        """
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Configuration file {filename} not found")
            return {}

def main():
    """
    Example usage of the DroneVisualizer
    """
    # Initialize visualizer
    visualizer = DroneVisualizer("stanford_data/archive")
    
    # Example: Visualize drone positions for bookstore location
    location = "bookstore"
    drone_positions = [
        (300, 200),
        (600, 400),
        (900, 600),
    ]
    fov_radius = 180
    
    # Check for overlaps
    overlaps = visualizer.check_overlaps(drone_positions, fov_radius)
    if overlaps:
        print(f"Warning: Overlapping FOVs detected between drones: {overlaps}")
    else:
        print("No overlapping FOVs detected")
    
    # Visualize positions
    visualizer.visualize_drone_positions(location, drone_positions, fov_radius)
    
    # Generate automatic non-overlapping positions
    # First, we need to get image dimensions
    ref_path = Path("stanford_data/archive/annotations/bookstore/video4/reference.jpg")
    if ref_path.exists():
        image = cv2.imread(str(ref_path))
        if image is not None:
            height, width = image.shape[:2]
            auto_positions = visualizer.generate_non_overlapping_positions(
                (height, width), 4, fov_radius)
            print(f"Auto-generated positions: {auto_positions}")
            
            # Visualize auto-generated positions
            visualizer.visualize_drone_positions(location, auto_positions, fov_radius)
    
    # Save configuration
    config = {
        'bookstore': drone_positions,
        'coupa': [(250, 250), (550, 250), (400, 500), (0,0),],
    }
    visualizer.save_drone_config(config, fov_radius)


if __name__ == "__main__":
    # main()
    location = "quad"
    drone_positions = [
        (500, 250),
        (300, 800),   # drone2
        (1150, 750),   # drone3
        (1500, 300)
    ]
    fov_radius = 210
    vis = DroneVisualizer("stanford_data/archive")
    vis.visualize_drone_positions(location, drone_positions, fov_radius)