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
                                fov_side: float, save_path: str = None):
        """
        Visualize drone positions and square FOVs on the reference image
        
        Args:
            location: Location name
            drone_positions: List of (x, y) positions for drones
            fov_side: FOV side length
            save_path: Optional path to save the visualization
        """
        # Load reference image
        ref_path = self.dataset_path / "annotations" / location / "video1" / "reference.jpg"
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
            half_side = fov_side / 2
            
            # Calculate square FOV bounds
            x_min = int(x - half_side)
            x_max = int(x + half_side)
            y_min = int(y - half_side)
            y_max = int(y + half_side)
            
            # Draw FOV square
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw drone position (center point)
            cv2.circle(vis_image, (int(x), int(y)), 8, color, -1)
            
            # Draw crosshair at center
            cv2.line(vis_image, (int(x-15), int(y)), (int(x+15), int(y)), color, 2)
            cv2.line(vis_image, (int(x), int(y-15)), (int(x), int(y+15)), color, 2)
            
            # Add drone label
            cv2.putText(vis_image, f'D{i+1}', (int(x+half_side/2), int(y-half_side/2)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Drone Positions and Square FOVs - {location}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def check_overlaps(self, drone_positions: List[Tuple[float, float]], 
                      fov_side: float) -> List[Tuple[int, int]]:
        """
        Check for overlapping square FOVs and return pairs of overlapping drones
        
        Args:
            drone_positions: List of (x, y) positions for drones
            fov_side: FOV side length
            
        Returns:
            List of tuples containing indices of overlapping drone pairs
        """
        overlaps = []
        half_side = fov_side / 2
        
        for i in range(len(drone_positions)):
            for j in range(i + 1, len(drone_positions)):
                pos1 = drone_positions[i]
                pos2 = drone_positions[j]
                
                # Calculate bounds for drone 1
                drone1_left = pos1[0] - half_side
                drone1_right = pos1[0] + half_side
                drone1_top = pos1[1] - half_side
                drone1_bottom = pos1[1] + half_side
                
                # Calculate bounds for drone 2
                drone2_left = pos2[0] - half_side
                drone2_right = pos2[0] + half_side
                drone2_top = pos2[1] - half_side
                drone2_bottom = pos2[1] + half_side
                
                # Check for overlap
                x_overlap = drone1_right > drone2_left and drone1_left < drone2_right
                y_overlap = drone1_bottom > drone2_top and drone1_top < drone2_bottom
                
                if x_overlap and y_overlap:
                    overlaps.append((i, j))
        
        return overlaps
    
    def generate_non_overlapping_positions(self, image_shape: Tuple[int, int], 
                                         num_drones: int, fov_side: float,
                                         max_attempts: int = 1000) -> List[Tuple[float, float]]:
        """
        Generate non-overlapping drone positions within image bounds
        
        Args:
            image_shape: (height, width) of the reference image
            num_drones: Number of drones to place
            fov_side: FOV side length
            max_attempts: Maximum attempts to generate valid positions
            
        Returns:
            List of (x, y) positions for drones
        """
        height, width = image_shape
        positions = []
        half_side = fov_side / 2
        
        for attempt in range(max_attempts):
            if len(positions) >= num_drones:
                break
            
            # Generate random position within image bounds (with margin for FOV)
            x = np.random.randint(int(half_side), int(width - half_side))
            y = np.random.randint(int(half_side), int(height - half_side))
            
            # Check if this position overlaps with existing positions
            valid = True
            for existing_pos in positions:
                # Calculate bounds for new position
                new_left = x - half_side
                new_right = x + half_side
                new_top = y - half_side
                new_bottom = y + half_side
                
                # Calculate bounds for existing position
                existing_left = existing_pos[0] - half_side
                existing_right = existing_pos[0] + half_side
                existing_top = existing_pos[1] - half_side
                existing_bottom = existing_pos[1] + half_side
                
                # Check for overlap
                x_overlap = new_right > existing_left and new_left < existing_right
                y_overlap = new_bottom > existing_top and new_top < existing_bottom
                
                if x_overlap and y_overlap:
                    valid = False
                    break
            
            if valid:
                positions.append((float(x), float(y)))
        
        if len(positions) < num_drones:
            print(f"Warning: Could only generate {len(positions)} non-overlapping positions out of {num_drones} requested")
        
        return positions
    
    def calculate_coverage_area(self, drone_positions: List[Tuple[float, float]], 
                              fov_side: float, image_shape: Tuple[int, int]) -> float:
        """
        Calculate the total coverage area of all drones
        
        Args:
            drone_positions: List of (x, y) positions for drones
            fov_side: FOV side length
            image_shape: (height, width) of the reference image
            
        Returns:
            Coverage percentage (0-100)
        """
        height, width = image_shape
        coverage_mask = np.zeros((height, width), dtype=np.uint8)
        half_side = fov_side / 2
        
        for x, y in drone_positions:
            # Calculate square bounds
            x_min = max(0, int(x - half_side))
            x_max = min(width, int(x + half_side))
            y_min = max(0, int(y - half_side))
            y_max = min(height, int(y + half_side))
            
            # Mark coverage area
            coverage_mask[y_min:y_max, x_min:x_max] = 255
        
        # Calculate coverage percentage
        covered_pixels = np.sum(coverage_mask == 255)
        total_pixels = height * width
        coverage_percentage = (covered_pixels / total_pixels) * 100
        
        return coverage_percentage
    
    def save_drone_config(self, drone_positions: Dict[str, List[Tuple[float, float]]], 
                         fov_side: float, filename: str = "drone_config_square.json"):
        """
        Save drone configuration to JSON file
        
        Args:
            drone_positions: Dictionary mapping location names to drone positions
            fov_side: FOV side length
            filename: Output filename
        """
        config = {
            "fov_side": fov_side,
            "fov_type": "square",
            "drone_positions": drone_positions
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Drone configuration saved to: {filename}")
    
    def load_drone_config(self, filename: str = "drone_config_square.json") -> Dict:
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
    
    def visualize_coverage_heatmap(self, location: str, drone_positions: List[Tuple[float, float]], 
                                  fov_side: float, save_path: str = None):
        """
        Visualize coverage heatmap showing overlapping areas
        
        Args:
            location: Location name
            drone_positions: List of (x, y) positions for drones
            fov_side: FOV side length
            save_path: Optional path to save the visualization
        """
        # Load reference image to get dimensions
        ref_path = self.dataset_path / "annotations" / location / "video0" / "reference.jpg"
        if not ref_path.exists():
            location_path = self.dataset_path / "annotations" / location
            for video_dir in location_path.iterdir():
                if video_dir.is_dir():
                    ref_path = video_dir / "reference.jpg"
                    if ref_path.exists():
                        break
        
        if not ref_path.exists():
            print(f"No reference image found for location: {location}")
            return
        
        image = cv2.imread(str(ref_path))
        if image is None:
            print(f"Could not load reference image: {ref_path}")
            return
        
        height, width = image.shape[:2]
        coverage_map = np.zeros((height, width), dtype=np.uint8)
        half_side = fov_side / 2
        
        # Create coverage map
        for x, y in drone_positions:
            x_min = max(0, int(x - half_side))
            x_max = min(width, int(x + half_side))
            y_min = max(0, int(y - half_side))
            y_max = min(height, int(y + half_side))
            
            coverage_map[y_min:y_max, x_min:x_max] += 1
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with drone positions
        vis_image = image.copy()
        for i, (x, y) in enumerate(drone_positions):
            color = self.colors[i % len(self.colors)]
            half_side = fov_side / 2
            x_min = int(x - half_side)
            x_max = int(x + half_side)
            y_min = int(y - half_side)
            y_max = int(y + half_side)
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.circle(vis_image, (int(x), int(y)), 8, color, -1)
        
        ax1.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Drone Positions - {location}')
        ax1.axis('off')
        
        # Coverage heatmap
        im = ax2.imshow(coverage_map, cmap='hot', interpolation='nearest')
        ax2.set_title(f'Coverage Heatmap - {location}')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, label='Coverage Count')
        
        # Calculate and display coverage stats
        coverage_percentage = self.calculate_coverage_area(drone_positions, fov_side, (height, width))
        overlapping_areas = np.sum(coverage_map > 1)
        total_covered = np.sum(coverage_map > 0)
        
        fig.suptitle(f'Coverage: {coverage_percentage:.1f}% | Overlapping pixels: {overlapping_areas} | Total covered: {total_covered}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()

def main():
    """
    Example usage of the DroneVisualizer with square FOVs
    """
    # Initialize visualizer
    visualizer = DroneVisualizer("stanford_data/archive")
    
    # Example: Visualize drone positions for bookstore location
    location = "deathCircle"
    drone_positions = [
        (650, 350),
        (1050, 1000),
        (650, 1500),
        (350, 1000),
    ]
    fov_side = 360  # Square side length
    
    # Check for overlaps
    overlaps = visualizer.check_overlaps(drone_positions, fov_side)
    if overlaps:
        print(f"Warning: Overlapping FOVs detected between drones: {overlaps}")
    else:
        print("No overlapping FOVs detected")
    
    # Visualize positions
    visualizer.visualize_drone_positions(location, drone_positions, fov_side)
    
    # Visualize coverage heatmap
    visualizer.visualize_coverage_heatmap(location, drone_positions, fov_side)
    
    # Generate automatic non-overlapping positions
    ref_path = Path("stanford_data/archive/annotations/bookstore/video4/reference.jpg")
    if ref_path.exists():
        image = cv2.imread(str(ref_path))
        if image is not None:
            height, width = image.shape[:2]
            auto_positions = visualizer.generate_non_overlapping_positions(
                (height, width), 4, fov_side)
            print(f"Auto-generated positions: {auto_positions}")
            
            # Calculate coverage
            coverage = visualizer.calculate_coverage_area(auto_positions, fov_side, (height, width))
            print(f"Coverage area: {coverage:.1f}%")
            
            # Visualize auto-generated positions
            visualizer.visualize_drone_positions(location, auto_positions, fov_side)
    
    # Save configuration
    config = {
        'bookstore': drone_positions,
        'quad': [(500, 250), (1000, 250), (1500, 250), (500, 750)],
    }
    visualizer.save_drone_config(config, fov_side)

if __name__ == "__main__":
    # Example usage for quad location
    # location = "quad"
    # drone_positions = [
    #     (500, 250),
    #     (1000, 250),
    #     (1500, 250),
    #     (500, 750),
    #     (1000, 750),
    #     (1500, 750)
    # ]
    # fov_side = 420  # Square side length
    location = "deathCircle"
    drone_positions = [
        (650, 350),
        (1050, 1000),
        (650, 1500),
        (350, 1000),

        ]
    fov_side = 420  # Square side length
    
    vis = DroneVisualizer("/Users/aakarshrai/Desktop/stanford_data/archive")
    vis.visualize_drone_positions(location, drone_positions, fov_side)
    
    # # Check for overlaps
    overlaps = vis.check_overlaps(drone_positions, fov_side)
    if overlaps:
        print(f"Overlapping FOVs detected between drones: {overlaps}")
    else:
        print("No overlapping FOVs detected")
    
    # # Show coverage heatmap
    # vis.visualize_coverage_heatmap(location, drone_positions, fov_side)