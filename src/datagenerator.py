import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

@dataclass
class OrbitingObject:
    """Represents an object orbiting around a center point"""
    id: int
    radius: float
    angular_velocity: float  # radians per time step
    initial_angle: float
    direction: int  # 1 for clockwise, -1 for counterclockwise
    
    def get_position(self, time_step: int) -> Tuple[float, float]:
        """Calculate x, y position at given time step"""
        angle = self.initial_angle + self.direction * self.angular_velocity * time_step
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        return x, y

class Camera:
    """Represents a camera monitoring a quadrant"""
    def __init__(self, quadrant: int, name: str):
        self.quadrant = quadrant  # 1: Q1 (x>0, y>0), 2: Q2 (x<0, y>0), 3: Q3 (x<0, y<0), 4: Q4 (x>0, y<0)
        self.name = name
        self.tracked_objects = set()
        self.events = []  # List of enter/exit events
    
    def is_in_quadrant(self, x: float, y: float) -> bool:
        """Check if position is in this camera's quadrant"""
        if self.quadrant == 1:
            return x >= 0 and y >= 0
        elif self.quadrant == 2:
            return x < 0 and y >= 0
        elif self.quadrant == 3:
            return x < 0 and y < 0
        elif self.quadrant == 4:
            return x >= 0 and y < 0
        return False
    
    def update(self, object_id: int, x: float, y: float, time_step: int):
        """Update camera tracking for an object"""
        in_quadrant = self.is_in_quadrant(x, y)
        was_tracked = object_id in self.tracked_objects
        
        if in_quadrant and not was_tracked:
            # Object enters quadrant
            self.tracked_objects.add(object_id)
            self.events.append({
                'time_step': time_step,
                'object_id': object_id,
                'event': 'enter',
                'camera': self.name,
                'quadrant': self.quadrant,
                'x': x,
                'y': y
            })
        elif not in_quadrant and was_tracked:
            # Object exits quadrant
            self.tracked_objects.remove(object_id)
            self.events.append({
                'time_step': time_step,
                'object_id': object_id,
                'event': 'exit',
                'camera': self.name,
                'quadrant': self.quadrant,
                'x': x,
                'y': y
            })

class OrbitSimulation:
    """Main simulation class"""
    def __init__(self, num_objects: int = 1000, time_steps: int = 1000):
        self.num_objects = num_objects
        self.time_steps = time_steps
        self.objects = []
        self.cameras = [
            Camera(1, "Camera_Q1"),
            Camera(2, "Camera_Q2"), 
            Camera(3, "Camera_Q3"),
            Camera(4, "Camera_Q4")
        ]
        self.position_data = []
        
        # Initialize objects with random parameters
        self._create_objects()
    
    def _create_objects(self):
        """Create orbiting objects with random parameters"""
        for i in range(self.num_objects):
            obj = OrbitingObject(
                id=i,
                radius=random.uniform(50, 500),  # Random radius between 50-500 units
                angular_velocity=random.uniform(0.01, 0.1),  # Random speed
                initial_angle=random.uniform(0, 2*np.pi),  # Random starting position
                direction=random.choice([1, -1])  # Random direction
            )
            self.objects.append(obj)
    
    def run_simulation(self):
        """Run the complete simulation"""
        print(f"Running simulation with {self.num_objects} objects for {self.time_steps} time steps...")
        
        for t in range(self.time_steps):
            if t % 100 == 0:
                print(f"Processing time step {t}/{self.time_steps}")
            
            for obj in self.objects:
                x, y = obj.get_position(t)
                
                # Record position data
                self.position_data.append({
                    'time_step': t,
                    'object_id': obj.id,
                    'x': x,
                    'y': y,
                    'radius': obj.radius,
                    'angular_velocity': obj.angular_velocity,
                    'direction': obj.direction
                })
                
                # Update all cameras
                for camera in self.cameras:
                    camera.update(obj.id, x, y, t)
        
        print("Simulation complete!")
    
    def get_position_dataset(self) -> pd.DataFrame:
        """Get the position dataset as a pandas DataFrame"""
        return pd.DataFrame(self.position_data)
    
    def get_camera_events(self) -> pd.DataFrame:
        """Get all camera events as a pandas DataFrame"""
        all_events = []
        for camera in self.cameras:
            all_events.extend(camera.events)
        return pd.DataFrame(all_events)
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        camera_events_df = self.get_camera_events()
        stats = {
            'total_objects': self.num_objects,
            'total_time_steps': self.time_steps,
            'total_position_records': len(self.position_data),
            'total_camera_events': len(camera_events_df),
            'events_per_camera': {}
        }
        
        for camera in self.cameras:
            camera_events = len(camera.events)
            enter_events = len([e for e in camera.events if e['event'] == 'enter'])
            exit_events = len([e for e in camera.events if e['event'] == 'exit'])
            
            stats['events_per_camera'][camera.name] = {
                'total_events': camera_events,
                'enter_events': enter_events,
                'exit_events': exit_events
            }
        
        return stats
    
    def save_datasets(self, base_filename: str = "orbit_simulation"):
        """Save datasets to CSV files"""
        # Save position data
        position_df = self.get_position_dataset()
        position_filename = f"{base_filename}_positions.csv"
        position_df.to_csv(position_filename, index=False)
        print(f"Position dataset saved to {position_filename}")
        
        # Save camera events
        events_df = self.get_camera_events()
        events_filename = f"{base_filename}_camera_events.csv"
        events_df.to_csv(events_filename, index=False)
        print(f"Camera events saved to {events_filename}")
        
        # Save statistics
        stats = self.get_statistics()
        stats_filename = f"{base_filename}_statistics.json"
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_filename}")
        
        return position_filename, events_filename, stats_filename
    
    def visualize_snapshot(self, time_step: int = 0, save_plot: bool = True):
        """Create a visualization of object positions at a specific time step"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Get positions at specified time step
        positions = []
        colors = []
        for obj in self.objects:
            x, y = obj.get_position(time_step)
            positions.append((x, y))
            
            # Color by quadrant
            if x >= 0 and y >= 0:
                colors.append('red')  # Q1
            elif x < 0 and y >= 0:
                colors.append('blue')  # Q2
            elif x < 0 and y < 0:
                colors.append('green')  # Q3
            else:
                colors.append('orange')  # Q4
        
        # Plot objects
        x_coords, y_coords = zip(*positions)
        ax.scatter(x_coords, y_coords, c=colors, alpha=0.6, s=20)
        
        # Add quadrant lines
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(250, 250, 'Q1\n(Camera 1)', ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        ax.text(-250, 250, 'Q2\n(Camera 2)', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3))
        ax.text(-250, -250, 'Q3\n(Camera 3)', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
        ax.text(250, -250, 'Q4\n(Camera 4)', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))
        
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)
        ax.set_aspect('equal')
        ax.set_title(f'Orbital Simulation - Time Step {time_step}\n{self.num_objects} Objects')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig(f'orbit_simulation_snapshot_t{time_step}.png', dpi=300, bbox_inches='tight')
            print(f"Visualization saved as orbit_simulation_snapshot_t{time_step}.png")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create and run simulation
    sim = OrbitSimulation(num_objects=100, time_steps=200)
    sim.run_simulation()
    
    # Get datasets
    position_data = sim.get_position_dataset()
    camera_events = sim.get_camera_events()
    
    # Print basic info
    print("\n=== SIMULATION RESULTS ===")
    print(f"Position dataset shape: {position_data.shape}")
    print(f"Camera events shape: {camera_events.shape}")
    
    print("\nFirst 5 position records:")
    print(position_data.head())
    
    print("\nFirst 5 camera events:")
    print(camera_events.head())
    
    # Print statistics
    stats = sim.get_statistics()
    print(f"\n=== STATISTICS ===")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Total time steps: {stats['total_time_steps']}")
    print(f"Total position records: {stats['total_position_records']}")
    print(f"Total camera events: {stats['total_camera_events']}")
    
    print("\nEvents per camera:")
    for camera_name, camera_stats in stats['events_per_camera'].items():
        print(f"  {camera_name}: {camera_stats['total_events']} events "
              f"({camera_stats['enter_events']} enters, {camera_stats['exit_events']} exits)")
    
    # Save datasets
    print("\n=== SAVING DATASETS ===")
    sim.save_datasets("orbit_simulation_1000_objects")
    
    # Create visualization
    print("\n=== CREATING VISUALIZATION ===")
    sim.visualize_snapshot(time_step=0)
    sim.visualize_snapshot(time_step=100)