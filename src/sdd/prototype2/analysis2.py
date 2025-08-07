#!/usr/bin/env python3
"""
Drone Trajectory Visualization Script

This script loads sequences from the OptimizedMultiAgentSequenceDataset and plots
trajectories on the reference drone images from the original dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from pathlib import Path
import argparse
import random
from typing import Dict, List, Tuple, Optional
import json
from newest_training_loop import OptimizedMultiAgentSequenceDataset

# Assuming the dataset class is imported from your module
# from your_dataset_module import OptimizedMultiAgentSequenceDataset

class DroneTrajectoryVisualizer:
    def __init__(self, dataset, original_dataset_root: str, output_dir: str = "trajectory_plots"):
        """
        Initialize the visualizer
        
        Args:
            dataset: Instance of OptimizedMultiAgentSequenceDataset
            original_dataset_root: Path to original dataset containing reference images
            output_dir: Directory to save visualization plots
        """
        self.dataset = dataset
        self.orig_root = Path(original_dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color schemes for different agents and trajectory parts
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
                      'olive', 'cyan', 'magenta', 'yellow', 'navy', 'teal', 'lime', 'indigo',
                      'violet', 'turquoise', 'gold', 'coral']
        
        # Different markers for past vs future
        self.past_marker = 'o'
        self.future_marker = 's'
        self.past_alpha = 0.8
        self.future_alpha = 0.6
        
    def find_reference_image(self, location: str, video: str) -> Optional[str]:
        """
        Find the reference image for a given location and video
        
        Args:
            location: Location name
            video: Video name
            
        Returns:
            Path to reference image or None if not found
        """
        video_dir = self.orig_root / "annotations" / location / video
        
        if not video_dir.exists():
            return None
            
        # Look for common image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for ext in image_extensions:
            # Try different naming patterns
            possible_names = [
                f"reference{ext}",
                f"ref{ext}",
                f"background{ext}",
                f"frame_000001{ext}",
                f"img_000001{ext}",
                f"{video}{ext}",
            ]
            
            for name in possible_names:
                img_path = video_dir / name
                if img_path.exists():
                    return str(img_path)
        
        # If no specific reference image found, look for any image in the directory
        for file in video_dir.iterdir():
            if file.suffix.lower() in image_extensions:
                return str(file)
                
        return None
    
    def denormalize_positions(self, positions: np.ndarray, sample: Dict) -> np.ndarray:
        """
        Denormalize positions using dataset's denormalization method
        
        Args:
            positions: Normalized positions array
            sample: Sample dictionary containing location and video info
            
        Returns:
            Denormalized positions
        """
        video_key = f"{sample['location']}_{sample['video']}"
        
        # Use the dataset's denormalization method
        if hasattr(self.dataset, 'denormalize_coordinates'):
            # Handle pad values properly
            mask = positions != self.dataset.pad_value
            denorm_positions = positions.copy()
            
            # Only denormalize non-padded values
            valid_positions = positions[mask].reshape(-1, 2)
            if len(valid_positions) > 0:
                denorm_valid = self.dataset.denormalize_coordinates(valid_positions, video_key)
                denorm_positions[mask] = denorm_valid.flatten()
            
            return denorm_positions
        else:
            return positions
    
    def plot_trajectory_on_image(self, sample: Dict, save_path: str = None, 
                                show_plot: bool = True, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot trajectories from a sample on the reference image
        
        Args:
            sample: Sample dictionary from dataset
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        # Find reference image
        ref_img_path = self.find_reference_image(sample['location'], sample['video'])
        
        if ref_img_path is None:
            print(f"Warning: No reference image found for {sample['location']}/{sample['video']}")
            # Create a blank image as fallback
            img_height, img_width = 480, 640  # Default size
            reference_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 128
        else:
            # Load reference image
            reference_img = cv2.imread(ref_img_path)
            if reference_img is None:
                print(f"Warning: Could not load image {ref_img_path}")
                img_height, img_width = 480, 640
                reference_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 128
            else:
                reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
                img_height, img_width = reference_img.shape[:2]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(reference_img)
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Flip y-axis for image coordinates
        
        # Get trajectory data
        past_positions = sample['past_positions']
        future_positions = sample['future_positions']
        agent_masks = sample['agent_masks']
        temporal_masks_past = sample['temporal_masks_past']
        temporal_masks_future = sample['temporal_masks_future']
        obs_masks = sample.get('obs_masks', np.ones_like(temporal_masks_past))
        agent_ids = sample.get('agent_ids', np.arange(len(agent_masks)))
        
        # Use original positions if available, otherwise denormalize
        if 'past_positions_orig' in sample and 'future_positions_orig' in sample:
            past_pos_plot = sample['past_positions_orig']
            future_pos_plot = sample['future_positions_orig']
        else:
            past_pos_plot = self.denormalize_positions(past_positions, sample)
            future_pos_plot = self.denormalize_positions(future_positions, sample)
        
        # Plot trajectories for each agent
        legend_elements = []
        
        for agent_idx in range(len(agent_masks)):
            if agent_masks[agent_idx] == 0:
                continue
                
            agent_id = agent_ids[agent_idx] if agent_idx < len(agent_ids) else agent_idx
            color = self.colors[agent_idx % len(self.colors)]
            
            # Get valid past positions
            past_mask = temporal_masks_past[agent_idx] > 0
            valid_past_pos = past_pos_plot[agent_idx][past_mask]
            
            # Filter out pad values
            if hasattr(self.dataset, 'pad_value'):
                valid_past_mask = ~np.any(valid_past_pos == self.dataset.pad_value, axis=1)
                valid_past_pos = valid_past_pos[valid_past_mask]
                past_obs = obs_masks[agent_idx][past_mask][valid_past_mask] if len(obs_masks.shape) > 1 else None
            else:
                past_obs = obs_masks[agent_idx][past_mask] if len(obs_masks.shape) > 1 else None
            
            # Get valid future positions
            future_mask = temporal_masks_future[agent_idx] > 0
            valid_future_pos = future_pos_plot[agent_idx][future_mask]
            
            # Filter out pad values
            if hasattr(self.dataset, 'pad_value'):
                valid_future_mask = ~np.any(valid_future_pos == self.dataset.pad_value, axis=1)
                valid_future_pos = valid_future_pos[valid_future_mask]
            
            # Plot past trajectory
            if len(valid_past_pos) > 0:
                # Plot trajectory line
                if len(valid_past_pos) > 1:
                    ax.plot(valid_past_pos[:, 0], valid_past_pos[:, 1], 
                           color=color, linewidth=2, alpha=self.past_alpha, linestyle='-')
                
                # Plot points with different markers for observed vs unobserved
                if past_obs is not None:
                    observed_pos = valid_past_pos[past_obs > 0]
                    unobserved_pos = valid_past_pos[past_obs == 0]
                    
                    if len(observed_pos) > 0:
                        ax.scatter(observed_pos[:, 0], observed_pos[:, 1], 
                                 c=color, marker=self.past_marker, s=50, alpha=self.past_alpha,
                                 edgecolors='black', linewidth=1, label=f'Agent {agent_id} (Past, Observed)')
                    
                    if len(unobserved_pos) > 0:
                        ax.scatter(unobserved_pos[:, 0], unobserved_pos[:, 1], 
                                 c=color, marker=self.past_marker, s=50, alpha=self.past_alpha//2,
                                 edgecolors='black', linewidth=1, linestyle='--',
                                 label=f'Agent {agent_id} (Past, Unobserved)')
                else:
                    ax.scatter(valid_past_pos[:, 0], valid_past_pos[:, 1], 
                             c=color, marker=self.past_marker, s=50, alpha=self.past_alpha,
                             edgecolors='black', linewidth=1)
            
            # Plot future trajectory
            if len(valid_future_pos) > 0:
                # Plot trajectory line
                if len(valid_future_pos) > 1:
                    ax.plot(valid_future_pos[:, 0], valid_future_pos[:, 1], 
                           color=color, linewidth=2, alpha=self.future_alpha, linestyle='--')
                
                # Plot future points
                ax.scatter(valid_future_pos[:, 0], valid_future_pos[:, 1], 
                         c=color, marker=self.future_marker, s=40, alpha=self.future_alpha,
                         edgecolors='black', linewidth=1)
                
                # Connect last past point to first future point
                if len(valid_past_pos) > 0:
                    ax.plot([valid_past_pos[-1, 0], valid_future_pos[0, 0]], 
                           [valid_past_pos[-1, 1], valid_future_pos[0, 1]], 
                           color=color, linewidth=1, alpha=0.5, linestyle=':')
            
            # Add to legend
            if len(valid_past_pos) > 0 or len(valid_future_pos) > 0:
                legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                                label=f'Agent {agent_id}'))
        
        # Customize plot
        ax.set_title(f'Trajectory Visualization\n{sample["location"]} - {sample["video"]}\n'
                    f'Frame: {sample.get("start_frame", "Unknown")} | '
                    f'Valid Agents: {sample.get("num_valid_agents", "Unknown")}')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add legend
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add frame subsample info if available
        if 'frame_subsample_rate' in sample:
            ax.text(0.02, 0.98, f'Frame subsample rate: {sample["frame_subsample_rate"]}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def visualize_random_samples(self, num_samples: int = 5, seed: int = 42):
        """
        Visualize random samples from the dataset
        
        Args:
            num_samples: Number of random samples to visualize
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Get random sample indices
        sample_indices = random.sample(range(len(self.dataset)), 
                                     min(num_samples, len(self.dataset)))
        
        print(f"Visualizing {len(sample_indices)} random samples...")
        
        for i, idx in enumerate(sample_indices):
            print(f"Processing sample {i+1}/{len(sample_indices)} (index {idx})...")
            
            try:
                sample = self.dataset[idx]
                
                # Create save path
                save_filename = f"trajectory_sample_{idx}_{sample['location']}_{sample['video']}.png"
                save_path = self.output_dir / save_filename
                
                # Plot trajectory
                self.plot_trajectory_on_image(sample, save_path=str(save_path), show_plot=False)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    def visualize_specific_sample(self, location: str, video: str, start_frame: int = None):
        """
        Visualize a specific sample by location and video
        
        Args:
            location: Location name
            video: Video name  
            start_frame: Specific start frame (optional)
        """
        print(f"Looking for samples from {location}/{video}...")
        
        found_samples = []
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                if (sample['location'] == location and sample['video'] == video and
                    (start_frame is None or sample.get('start_frame') == start_frame)):
                    found_samples.append((idx, sample))
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                continue
        
        if not found_samples:
            print(f"No samples found for {location}/{video}")
            return
        
        print(f"Found {len(found_samples)} matching samples")
        
        # Visualize all found samples
        for i, (idx, sample) in enumerate(found_samples):
            save_filename = f"trajectory_{location}_{video}_idx{idx}_frame{sample.get('start_frame', 'unknown')}.png"
            save_path = self.output_dir / save_filename
            
            print(f"Visualizing sample {i+1}/{len(found_samples)}: {save_filename}")
            self.plot_trajectory_on_image(sample, save_path=str(save_path), show_plot=True)


# Configuration
CONFIG = {
    'drone_data_root': '/Users/aakarshrai/Desktop/square_stanford_data',  # UPDATE THIS PATH
    'original_dataset_root': '/Users/aakarshrai/Desktop/stanford_data/archive/',  # UPDATE THIS PATH
    'cache_dir': '/Users/aakarshrai/Downloads/new_cache_64',
    'output_dir': 'trajectory_plots',
    'classes': ['Pedestrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus'],
    'num_samples': 5,
    'location': None,  # Set to specific location name or None for random
    'video': None,     # Set to specific video name or None for random
    'start_frame': None,  # Set to specific frame or None
    'T_past': 10,
    'T_future': 20,
    'frame_subsample': 12,
    'seed': 42,
    'num_workers': 4,
    'lazy_loading': True
}


def main():
    print("Initializing dataset...")
    
    # Initialize dataset (you'll need to import your dataset class)
    try:
        dataset = OptimizedMultiAgentSequenceDataset(
            drone_data_root=CONFIG['drone_data_root'],
            original_dataset_root=CONFIG['original_dataset_root'],
            classes=CONFIG['classes'],
            T_past=CONFIG['T_past'],
            T_future=CONFIG['T_future'],
            cache_dir=CONFIG['cache_dir'],
            frame_subsample=CONFIG['frame_subsample'],
            lazy_loading=CONFIG['lazy_loading'],
            num_workers=CONFIG['num_workers'],
            max_agents=64
        )
        
        print(f"Dataset initialized with {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        print("Make sure to import the OptimizedMultiAgentSequenceDataset class")
        return
    
    # Initialize visualizer
    visualizer = DroneTrajectoryVisualizer(dataset, CONFIG['original_dataset_root'], CONFIG['output_dir'])
    
    # Visualize samples
    if CONFIG['location'] and CONFIG['video']:
        # Visualize specific location/video
        visualizer.visualize_specific_sample(CONFIG['location'], CONFIG['video'], CONFIG['start_frame'])
    else:
        # Visualize random samples
        visualizer.visualize_random_samples(CONFIG['num_samples'], CONFIG['seed'])
    
    print(f"Visualization complete! Check {CONFIG['output_dir']} for saved plots.")


if __name__ == "__main__":
    main()