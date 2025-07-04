# drone_vision_visualizer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from dataclasses import dataclass

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

@dataclass
class DroneConfig:
    """Configuration for a drone (recreated for visualization)"""
    drone_id: str
    center_x: float
    center_y: float
    radius: float
    fov_angle: float = 360.0
    facing_direction: float = 0.0
    height: float = 50.0

class DroneVisionVisualizer:
    """Visualize drone observations as videos"""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color mapping for agent types
        self.agent_colors = {
            'VEHICLE': '#FF6B6B',
            'PEDESTRIAN': '#4ECDC4', 
            'CYCLIST': '#45B7D1',
            'MOTORCYCLIST': '#FFA726',
            'BUS': '#AB47BC',
            'UNKNOWN': '#78909C'
        }
        
        # Drone colors
        self.drone_colors = {
            'center': '#FFD700',
            'left': '#FF69B4', 
            'right': '#00CED1',
            'top': '#32CD32',
            'bottom': '#FF4500'
        }
    
    def load_scenario_data(self, scenario_id: str) -> Tuple[Dict, List[DroneConfig]]:
        """Load all drone observations and analytics for a scenario"""
        
        # Load analytics to get drone configurations
        analytics_path = self.data_dir / f'analytics_{scenario_id}.json'
        if not analytics_path.exists():
            raise FileNotFoundError(f"Analytics file not found: {analytics_path}")
        
        with open(analytics_path, 'r') as f:
            analytics = json.load(f)
        
        # Reconstruct drone configurations
        drone_configs = []
        for drone_id, config in analytics['drone_configs'].items():
            drone_configs.append(DroneConfig(
                drone_id=drone_id,
                center_x=config['center_x_global'],
                center_y=config['center_y_global'],
                radius=config['radius'],
                facing_direction=config['facing_direction'],
                height=config['height']
            ))
        
        # Load observation data for each drone
        all_observations = {}
        for drone_id in analytics['drone_configs'].keys():
            csv_path = self.data_dir / f'{drone_id}_observations_{scenario_id}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_observations[drone_id] = df
            else:
                print(f"Warning: No observation file found for {drone_id}")
                all_observations[drone_id] = pd.DataFrame()
        
        return all_observations, drone_configs
    
    def create_bird_eye_view(self, scenario_id: str, save_video: bool = True) -> None:
        """Create bird's eye view video showing all drones and their observations"""
        
        print(f"Creating bird's eye view for scenario {scenario_id}...")
        
        # Load data
        all_observations, drone_configs = self.load_scenario_data(scenario_id)
        
        # Get all unique frames
        all_frames = set()
        for obs_df in all_observations.values():
            if not obs_df.empty:
                all_frames.update(obs_df['frame'].unique())
        
        if not all_frames:
            print("No observations found!")
            return
        
        frames = sorted(all_frames)
        
        # Calculate scene bounds
        all_x_global = []
        all_y_global = []
        for obs_df in all_observations.values():
            if not obs_df.empty:
                all_x_global.extend(obs_df['x_global'].values)
                all_y_global.extend(obs_df['y_global'].values)
        
        # Add drone positions to bounds
        for drone in drone_configs:
            all_x_global.extend([drone.center_x - drone.radius, drone.center_x + drone.radius])
            all_y_global.extend([drone.center_y - drone.radius, drone.center_y + drone.radius])
        
        x_min, x_max = min(all_x_global), max(all_x_global)
        y_min, y_max = min(all_y_global), max(all_y_global)
        
        # Add padding
        padding = max(x_max - x_min, y_max - y_min) * 0.1
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        def animate(frame_idx):
            ax.clear()
            
            current_frame = frames[frame_idx]
            
            # Set up plot
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.set_title(f'Drone Coverage - Scenario {scenario_id} - Frame {current_frame}', 
                        fontsize=16, fontweight='bold', color='white')
            ax.set_xlabel('X (meters)', fontsize=12, color='white')
            ax.set_ylabel('Y (meters)', fontsize=12, color='white')
            
            # Draw drone FOV circles
            for drone in drone_configs:
                circle = patches.Circle(
                    (drone.center_x, drone.center_y), 
                    drone.radius,
                    fill=False, 
                    edgecolor=self.drone_colors.get(drone.drone_id, '#FFFFFF'),
                    linewidth=2,
                    alpha=0.6,
                    linestyle='--'
                )
                ax.add_patch(circle)
                
                # Add drone center
                ax.plot(drone.center_x, drone.center_y, 
                       marker='s', markersize=12, 
                       color=self.drone_colors.get(drone.drone_id, '#FFFFFF'),
                       markeredgecolor='black', markeredgewidth=2)
                
                # Add drone label
                ax.text(drone.center_x, drone.center_y + drone.radius + 5, 
                       f'Drone {drone.drone_id.title()}',
                       ha='center', va='bottom', 
                       fontsize=10, fontweight='bold',
                       color=self.drone_colors.get(drone.drone_id, '#FFFFFF'))
            
            # Plot observations for current frame
            legend_elements = []
            agent_counts = {}
            
            for drone_id, obs_df in all_observations.items():
                if obs_df.empty:
                    continue
                    
                frame_obs = obs_df[obs_df['frame'] == current_frame]
                
                for _, obs in frame_obs.iterrows():
                    agent_type = obs['agent_type']
                    color = self.agent_colors.get(agent_type, '#FFFFFF')
                    
                    # Different marker for focal agent
                    if obs['is_focal']:
                        marker = '*'
                        size = 150
                        edgecolor = 'yellow'
                        linewidth = 3
                    else:
                        marker = 'o'
                        size = 80
                        edgecolor = 'white'
                        linewidth = 1
                    
                    ax.scatter(obs['x_global'], obs['y_global'], 
                             c=color, s=size, marker=marker,
                             edgecolors=edgecolor, linewidths=linewidth,
                             alpha=0.8, zorder=10)
                    
                    # Count agents by type
                    agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
            
            # Create legend
            for agent_type, color in self.agent_colors.items():
                if agent_type in agent_counts:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=10,
                                                    label=f'{agent_type} ({agent_counts[agent_type]})'))
            
            # Add focal agent to legend
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                            markerfacecolor='yellow', markersize=15,
                                            label='Focal Agent'))
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', 
                         bbox_to_anchor=(1, 1), framealpha=0.8)
            
            # Add statistics
            total_obs = sum(len(obs_df[obs_df['frame'] == current_frame]) 
                           for obs_df in all_observations.values())
            
            stats_text = f'Frame: {current_frame}\nTotal Observations: {total_obs}\nActive Drones: {len([d for d in drone_configs])}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                   color='white')
            
            ax.grid(True, alpha=0.3)
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(frames), interval=200, repeat=True)
        
        if save_video:
            video_path = self.output_dir / f'bird_eye_view_{scenario_id}.mp4'
            print(f"Saving video to {video_path}...")
            anim.save(str(video_path), writer='ffmpeg', fps=5, dpi=100)
            print(f"✓ Bird's eye view saved!")
        
        plt.tight_layout()
        plt.show()
    
    def create_drone_pov_video(self, scenario_id: str, drone_id: str, save_video: bool = True) -> None:
        """Create first-person view video from a specific drone's perspective"""
        
        print(f"Creating POV video for drone {drone_id} in scenario {scenario_id}...")
        
        # Load data
        all_observations, drone_configs = self.load_scenario_data(scenario_id)
        
        # Get specific drone config
        drone_config = next((d for d in drone_configs if d.drone_id == drone_id), None)
        if not drone_config:
            print(f"Drone {drone_id} not found!")
            return
        
        # Get observations for this drone
        if drone_id not in all_observations or all_observations[drone_id].empty:
            print(f"No observations found for drone {drone_id}")
            return
        
        obs_df = all_observations[drone_id]
        frames = sorted(obs_df['frame'].unique())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        def animate(frame_idx):
            ax.clear()
            
            current_frame = frames[frame_idx]
            frame_obs = obs_df[obs_df['frame'] == current_frame]
            
            # Set up plot (local coordinates, drone at center)
            ax.set_xlim(-drone_config.radius * 1.1, drone_config.radius * 1.1)
            ax.set_ylim(-drone_config.radius * 1.1, drone_config.radius * 1.1)
            ax.set_aspect('equal')
            ax.set_title(f'Drone {drone_id.title()} POV - Scenario {scenario_id} - Frame {current_frame}', 
                        fontsize=16, fontweight='bold', color='white')
            ax.set_xlabel('X (meters from drone)', fontsize=12, color='white')
            ax.set_ylabel('Y (meters from drone)', fontsize=12, color='white')
            
            # Draw drone FOV circle
            fov_circle = patches.Circle((0, 0), drone_config.radius, 
                                      fill=False, edgecolor='yellow', 
                                      linewidth=3, alpha=0.8)
            ax.add_patch(fov_circle)
            
            # Draw drone at center
            ax.plot(0, 0, marker='s', markersize=15, color='yellow',
                   markeredgecolor='black', markeredgewidth=2)
            ax.text(0, -drone_config.radius * 0.15, 'DRONE', 
                   ha='center', va='top', fontsize=12, fontweight='bold', color='yellow')
            
            # Plot observed agents
            legend_elements = []
            agent_counts = {}
            
            for _, obs in frame_obs.iterrows():
                agent_type = obs['agent_type']
                color = self.agent_colors.get(agent_type, '#FFFFFF')
                
                # Different marker for focal agent
                if obs['is_focal']:
                    marker = '*'
                    size = 200
                    edgecolor = 'yellow'
                    linewidth = 3
                else:
                    marker = 'o'
                    size = 100
                    edgecolor = 'white'
                    linewidth = 1
                
                ax.scatter(obs['x_local'], obs['y_local'], 
                         c=color, s=size, marker=marker,
                         edgecolors=edgecolor, linewidths=linewidth,
                         alpha=0.8, zorder=10)
                
                # Add distance label
                ax.text(obs['x_local'], obs['y_local'] + 3, 
                       f"{obs['distance']:.1f}m", 
                       ha='center', va='bottom', fontsize=8, color='white')
                
                # Count agents
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
            
            # Create legend
            for agent_type, color in self.agent_colors.items():
                if agent_type in agent_counts:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=10,
                                                    label=f'{agent_type} ({agent_counts[agent_type]})'))
            
            if any(frame_obs['is_focal']):
                legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                                markerfacecolor='yellow', markersize=15,
                                                label='Focal Agent'))
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', 
                         bbox_to_anchor=(1, 1), framealpha=0.8)
            
            # Add compass/direction indicator
            compass_x, compass_y = drone_config.radius * 0.8, drone_config.radius * 0.8
            ax.arrow(compass_x, compass_y, 0, drone_config.radius * 0.1, 
                    head_width=drone_config.radius * 0.03, head_length=drone_config.radius * 0.02,
                    fc='red', ec='red')
            ax.text(compass_x, compass_y + drone_config.radius * 0.15, 'N', 
                   ha='center', va='bottom', fontsize=12, color='red', fontweight='bold')
            
            # Add statistics
            stats_text = f'Frame: {current_frame}\nObservations: {len(frame_obs)}\nRadius: {drone_config.radius:.1f}m'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                   color='white')
            
            ax.grid(True, alpha=0.3)
            
            # Add range circles
            for r in [drone_config.radius * 0.25, drone_config.radius * 0.5, drone_config.radius * 0.75]:
                range_circle = patches.Circle((0, 0), r, fill=False, 
                                            edgecolor='gray', linewidth=1, alpha=0.3)
                ax.add_patch(range_circle)
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(frames), interval=200, repeat=True)
        
        if save_video:
            video_path = self.output_dir / f'drone_pov_{drone_id}_{scenario_id}.mp4'
            print(f"Saving video to {video_path}...")
            anim.save(str(video_path), writer='ffmpeg', fps=5, dpi=100)
            print(f"✓ Drone POV video saved!")
        
        plt.tight_layout()
        plt.show()
    
    def create_multi_drone_split_view(self, scenario_id: str, save_video: bool = True) -> None:
        """Create split-screen view showing multiple drone perspectives"""
        
        print(f"Creating multi-drone split view for scenario {scenario_id}...")
        
        # Load data
        all_observations, drone_configs = self.load_scenario_data(scenario_id)
        
        # Filter drones that have observations
        active_drones = [d for d in drone_configs 
                        if d.drone_id in all_observations and not all_observations[d.drone_id].empty]
        
        if not active_drones:
            print("No active drones found!")
            return
        
        # Get all frames
        all_frames = set()
        for drone in active_drones:
            obs_df = all_observations[drone.drone_id]
            all_frames.update(obs_df['frame'].unique())
        
        frames = sorted(all_frames)
        
        # Create subplot grid
        n_drones = len(active_drones)
        cols = min(3, n_drones)
        rows = (n_drones + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        if n_drones == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        def animate(frame_idx):
            current_frame = frames[frame_idx]
            
            for i, drone in enumerate(active_drones):
                ax = axes_flat[i]
                ax.clear()
                
                obs_df = all_observations[drone.drone_id]
                frame_obs = obs_df[obs_df['frame'] == current_frame]
                
                # Set up subplot
                ax.set_xlim(-drone.radius * 1.1, drone.radius * 1.1)
                ax.set_ylim(-drone.radius * 1.1, drone.radius * 1.1)
                ax.set_aspect('equal')
                ax.set_title(f'Drone {drone.drone_id.title()}', 
                           fontsize=12, fontweight='bold', color='white')
                
                # Draw FOV circle
                fov_circle = patches.Circle((0, 0), drone.radius, 
                                          fill=False, edgecolor=self.drone_colors.get(drone.drone_id, 'yellow'), 
                                          linewidth=2, alpha=0.8)
                ax.add_patch(fov_circle)
                
                # Draw drone
                ax.plot(0, 0, marker='s', markersize=8, 
                       color=self.drone_colors.get(drone.drone_id, 'yellow'),
                       markeredgecolor='black', markeredgewidth=1)
                
                # Plot observations
                for _, obs in frame_obs.iterrows():
                    agent_type = obs['agent_type']
                    color = self.agent_colors.get(agent_type, '#FFFFFF')
                    
                    if obs['is_focal']:
                        marker = '*'
                        size = 80
                        edgecolor = 'yellow'
                        linewidth = 2
                    else:
                        marker = 'o'
                        size = 40
                        edgecolor = 'white'
                        linewidth = 1
                    
                    ax.scatter(obs['x_local'], obs['y_local'], 
                             c=color, s=size, marker=marker,
                             edgecolors=edgecolor, linewidths=linewidth,
                             alpha=0.8)
                
                # Add observation count
                ax.text(0.02, 0.98, f'{len(frame_obs)} obs', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
                
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_drones, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            # Add main title
            fig.suptitle(f'Multi-Drone View - Scenario {scenario_id} - Frame {current_frame}', 
                        fontsize=16, fontweight='bold', color='white')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(frames), interval=200, repeat=True)
        
        if save_video:
            video_path = self.output_dir / f'multi_drone_view_{scenario_id}.mp4'
            print(f"Saving video to {video_path}...")
            anim.save(str(video_path), writer='ffmpeg', fps=5, dpi=100)
            print(f"✓ Multi-drone view saved!")
        
        plt.tight_layout()
        plt.show()
    
    def generate_all_videos(self, scenario_id: str) -> None:
        """Generate all types of videos for a scenario"""
        
        print(f"Generating all videos for scenario {scenario_id}...")
        
        # Bird's eye view
        self.create_bird_eye_view(scenario_id, save_video=True)
        
        # Multi-drone split view
        self.create_multi_drone_split_view(scenario_id, save_video=True)
        
        # Individual drone POVs
        _, drone_configs = self.load_scenario_data(scenario_id)
        for drone in drone_configs:
            self.create_drone_pov_video(scenario_id, drone.drone_id, save_video=True)
        
        print(f"✓ All videos generated for scenario {scenario_id}")

# ────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Configuration
    DATA_DIR = Path("src/argoverse2/enhanced_drone_datasets_final")
    OUTPUT_DIR = Path("src/argoverse2/drone_videos")
    
    # Create visualizer
    visualizer = DroneVisionVisualizer(DATA_DIR, OUTPUT_DIR)
    
    # Example usage - replace with your actual scenario ID
    #scenario_id = "001d7c9e-4f80-4d1d-9ad9-5b6432c4fb36"
    scenario_id = "001d7c9e-480-4d1d-9ad9-5b6432c4fb36"  # Example scenario ID
    
    # Generate specific video types
    visualizer.create_bird_eye_view(scenario_id)
    #visualizer.create_drone_pov_video(scenario_id, "center")
    #visualizer.create_multi_drone_split_view(scenario_id)
    
    # Or generate all videos at once
    # visualizer.generate_all_videos(scenario_id)
    
    print("Video generation complete!")
    print(f"Videos saved to: {OUTPUT_DIR}")
    print("\nAvailable methods:")
    print("1. create_bird_eye_view() - Overview of all drones and agents")
    print("2. create_drone_pov_video() - First-person view from specific drone")
    print("3. create_multi_drone_split_view() - Split-screen view of all drones")
    print("4. generate_all_videos() - Generate all video types")
# Usage example
# if __name__ == "__main__":
#     # Set your CSV directory path
#     csv_dir = Path("src/argoverse2/enhanced_drone_datasets_final")

#     # Create visualizer
#     visualizer = CSVFOVVisualizer(csv_dir)
    
#     # Replace with your actual scenario ID
#     scenario_id = "001d7c9e-4f80-4d1d-9ad9-5b6432c4fb36"  # Example scenario ID
#     # scenario_id = "00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff"
    
#     # Visualize specific frame
#     visualizer.visualize_fov(scenario_id, frame=0, save_path="drone_fov_frame0.png")
    
#     # Visualize all frames summary
#     visualizer.visualize_all_frames_summary(scenario_id, save_path="drone_fov_summary.png")