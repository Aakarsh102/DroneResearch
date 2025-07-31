import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import argparse

# Import your existing classes (assuming they're in the same directory or importable)
# You may need to adjust these imports based on your file structure
from training_loop3 import OptimizedMultiAgentSequenceDataset, collate_fn
from new_model import GraphInteractionModel

class TrajectoryVisualizer:
    """Visualize trajectory predictions on reference images"""
    
    def __init__(self, model_checkpoint_path, config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        
        self.model = GraphInteractionModel(
            num_classes=len(checkpoint['classes']),
            locations=checkpoint['locations'],
            d_model=16,
            nhead=8,
            num_layers=3,
            T_past=config['T_past'],
            T_future=config['T_future'],
            max_agents=config['max_agents']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.classes = checkpoint['classes']
        self.locations = checkpoint['locations']
        
        # Color scheme for visualization
        self.colors = {
            'past': '#2E86C1',        # Blue for past trajectory
            'predicted': '#E74C3C',   # Red for predicted trajectory
            'ground_truth': '#27AE60', # Green for ground truth future
            'observation': '#F39C12',  # Orange for observed positions
            'unobserved': '#85C1E9'   # Light blue for unobserved positions
        }
        
    def load_reference_image(self, location, video, original_dataset_root):
        """Load reference image for a video"""
        image_path = Path(original_dataset_root) / "annotations" / location / video / "reference.jpg"
        
        if not image_path.exists():
            print(f"Warning: Reference image not found at {image_path}")
            return None
            
        try:
            # Load image using OpenCV (BGR format)
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load image at {image_path}")
                return None
            # Convert BGR to RGB for matplotlib
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def denormalize_positions(self, positions, location, video, dataset):
        """Denormalize positions using dataset statistics"""
        if not dataset.normalize_positions or dataset.video_stats is None:
            return positions
            
        video_key = f"{location}_{video}"
        if video_key not in dataset.video_stats:
            return positions
            
        stats = dataset.video_stats[video_key]
        mean = stats['mean']
        std = stats['std']
        
        # Denormalize: x = (x_norm * std) + mean
        denormalized = positions * std + mean
        return denormalized
    
    def visualize_sequence(self, batch, predictions, idx, dataset, output_dir, sequence_name):
        """Visualize a single sequence"""
        # Extract data for specific sequence in batch
        location = batch['location'][idx] if isinstance(batch['location'], list) else batch['location'][idx].item()
        video = batch['video'][idx] if isinstance(batch['video'], list) else batch['video'][idx].item()
        
        # Load reference image
        reference_img = self.load_reference_image(location, video, dataset.orig_root)
        
        # Extract positions and masks for this sequence
        past_pos = batch['past_positions_orig'][idx].cpu().numpy()  # (max_agents, T_past, 2)
        future_pos_true = batch['future_positions_orig'][idx].cpu().numpy()  # (max_agents, T_future, 2)
        future_pos_pred = predictions['future_positions_mu'][idx].cpu().numpy()  # (max_agents, T_future, 2)
        
        obs_mask = batch['obs_masks'][idx].cpu().numpy()  # (max_agents, T_past)
        temporal_mask_past = batch['temporal_masks_past'][idx].cpu().numpy()  # (max_agents, T_past)
        temporal_mask_future = batch['temporal_masks_future'][idx].cpu().numpy()  # (max_agents, T_future)
        agent_mask = batch['agent_masks'][idx].cpu().numpy()  # (max_agents,)
        agent_ids = batch['agent_ids'][idx].cpu().numpy()  # (max_agents,)
        agent_labels = batch['agent_labels'][idx].cpu().numpy()  # (max_agents,)
        
        # Denormalize predicted positions (they might be normalized)
        if dataset.normalize_positions:
            # Reshape for denormalization
            future_pos_pred_reshaped = future_pos_pred.reshape(-1, 2)
            future_pos_pred_denorm = self.denormalize_positions(
                future_pos_pred_reshaped, location, video, dataset
            ).reshape(future_pos_pred.shape)
        else:
            future_pos_pred_denorm = future_pos_pred
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Show reference image if available
        if reference_img is not None:
            ax.imshow(reference_img, alpha=0.7)
            img_height, img_width = reference_img.shape[:2]
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0)  # Flip y-axis for image coordinates
        else:
            # If no image, set reasonable limits based on trajectory data
            all_pos = np.concatenate([
                past_pos[agent_mask > 0].reshape(-1, 2),
                future_pos_true[agent_mask > 0].reshape(-1, 2),
                future_pos_pred_denorm[agent_mask > 0].reshape(-1, 2)
            ], axis=0)
            
            if len(all_pos) > 0:
                padding = 50
                ax.set_xlim(all_pos[:, 0].min() - padding, all_pos[:, 0].max() + padding)
                ax.set_ylim(all_pos[:, 1].min() - padding, all_pos[:, 1].max() + padding)
        
        # Plot trajectories for each valid agent
        valid_agents = np.where(agent_mask > 0)[0]
        
        for agent_idx in valid_agents:
            agent_id = agent_ids[agent_idx]
            if agent_id == -1:  # Skip invalid agents
                continue
                
            # Get agent class
            class_idx = agent_labels[agent_idx]
            class_name = self.classes[class_idx] if 0 <= class_idx < len(self.classes) else "Unknown"
            
            # Past trajectory
            past_valid = temporal_mask_past[agent_idx] > 0
            if np.any(past_valid):
                past_traj = past_pos[agent_idx][past_valid]
                obs_valid = obs_mask[agent_idx][past_valid]
                
                # Plot past trajectory with different colors for observed/unobserved
                for i in range(len(past_traj)):
                    color = self.colors['observation'] if obs_valid[i] > 0 else self.colors['unobserved']
                    ax.scatter(past_traj[i, 0], past_traj[i, 1], 
                             c=color, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
                
                # Connect past points
                if len(past_traj) > 1:
                    ax.plot(past_traj[:, 0], past_traj[:, 1], 
                           color=self.colors['past'], linewidth=2, alpha=0.7, linestyle='-')
            
            # Future ground truth trajectory
            future_valid = temporal_mask_future[agent_idx] > 0
            if np.any(future_valid):
                future_traj_true = future_pos_true[agent_idx][future_valid]
                
                # Plot ground truth future
                ax.plot(future_traj_true[:, 0], future_traj_true[:, 1], 
                       color=self.colors['ground_truth'], linewidth=3, alpha=0.8, 
                       linestyle='-', label='Ground Truth' if agent_idx == valid_agents[0] else "")
                
                # Mark ground truth points
                ax.scatter(future_traj_true[:, 0], future_traj_true[:, 1], 
                         c=self.colors['ground_truth'], s=25, alpha=0.8, 
                         edgecolors='black', linewidth=0.5)
            
            # Future predicted trajectory
            if np.any(future_valid):
                future_traj_pred = future_pos_pred_denorm[agent_idx][future_valid]
                
                # Plot predicted future
                ax.plot(future_traj_pred[:, 0], future_traj_pred[:, 1], 
                       color=self.colors['predicted'], linewidth=3, alpha=0.8, 
                       linestyle='--', label='Predicted' if agent_idx == valid_agents[0] else "")
                
                # Mark predicted points
                ax.scatter(future_traj_pred[:, 0], future_traj_pred[:, 1], 
                         c=self.colors['predicted'], s=25, alpha=0.8, 
                         edgecolors='black', linewidth=0.5, marker='s')
            
            # Connect past to future (both ground truth and predicted)
            if np.any(past_valid) and np.any(future_valid):
                last_past = past_pos[agent_idx][past_valid][-1]
                first_future_true = future_pos_true[agent_idx][future_valid][0]
                first_future_pred = future_pos_pred_denorm[agent_idx][future_valid][0]
                
                # Connection lines
                ax.plot([last_past[0], first_future_true[0]], 
                       [last_past[1], first_future_true[1]], 
                       color=self.colors['ground_truth'], linewidth=2, alpha=0.5, linestyle=':')
                
                ax.plot([last_past[0], first_future_pred[0]], 
                       [last_past[1], first_future_pred[1]], 
                       color=self.colors['predicted'], linewidth=2, alpha=0.5, linestyle=':')
            
            # Add agent ID and class annotation
            if np.any(past_valid):
                last_pos = past_pos[agent_idx][past_valid][-1]
                ax.annotate(f'ID:{agent_id}\n{class_name}', 
                           xy=(last_pos[0], last_pos[1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Customize plot
        ax.set_title(f'Trajectory Prediction - {location}/{video}\n'
                    f'Frame Subsample: {batch["frame_subsample_rate"][idx]} | '
                    f'Valid Agents: {len(valid_agents)}', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors['past'], linewidth=2, label='Past Trajectory'),
            plt.Line2D([0], [0], color=self.colors['ground_truth'], linewidth=3, label='Ground Truth Future'),
            plt.Line2D([0], [0], color=self.colors['predicted'], linewidth=3, linestyle='--', label='Predicted Future'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['observation'], 
                      markersize=8, label='Observed Position'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['unobserved'], 
                      markersize=8, label='Unobserved Position')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        output_path = Path(output_dir) / f'{sequence_name}.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {output_path}")
        
        return {
            'location': location,
            'video': video,
            'num_agents': len(valid_agents),
            'agent_ids': agent_ids[valid_agents].tolist(),
            'agent_classes': [self.classes[agent_labels[i]] if 0 <= agent_labels[i] < len(self.classes) 
                            else "Unknown" for i in valid_agents]
        }
    
    def create_visualizations(self, dataset, num_sequences=8, output_dir='trajectory_visualizations'):
        """Create visualizations for multiple sequences"""
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=num_sequences,
            shuffle=True,
            num_workers=0,  # Use 0 for visualization to avoid multiprocessing issues
            collate_fn=collate_fn,
            drop_last=False
        )
        
        print(f"Creating visualizations for {num_sequences} sequences...")
        
        # Get one batch
        batch = next(iter(dataloader))
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        # Generate predictions
        print("Generating predictions...")
        with torch.no_grad():
            predictions = self.model(batch, use_teacher_forcing=False)
        
        # Create visualizations
        sequence_info = []
        actual_sequences = min(num_sequences, batch['past_positions'].size(0))
        
        for idx in range(actual_sequences):
            sequence_name = f'sequence_{idx+1:02d}'
            info = self.visualize_sequence(
                batch, predictions, idx, dataset, output_dir, sequence_name
            )
            sequence_info.append(info)
        
        # Save sequence information
        info_path = output_dir / 'sequence_info.json'
        with open(info_path, 'w') as f:
            json.dump(sequence_info, f, indent=2)
        
        print(f"\nCompleted! Created {actual_sequences} visualizations in {output_dir}")
        print(f"Sequence information saved to {info_path}")
        
        # Print summary
        print("\nSequence Summary:")
        for i, info in enumerate(sequence_info):
            print(f"  Sequence {i+1}: {info['location']}/{info['video']} - "
                  f"{info['num_agents']} agents - Classes: {', '.join(set(info['agent_classes']))}")


def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory predictions')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_config', type=str, 
                       help='Path to data configuration JSON file')
    parser.add_argument('--output_dir', type=str, default='trajectory_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_sequences', type=int, default=8,
                       help='Number of sequences to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Default configuration (modify as needed)
    default_config = {
        'drone_data_root': "/Users/aakarshrai/Desktop/square_stanford_data",
        'original_dataset_root': "/Users/aakarshrai/Desktop/stanford_data/archive",
        'cache_dir': "new_cache_16",
        'batch_size': 8,
        'num_workers': 0,
        'max_agents': 128,
        'T_past': 10,
        'T_future': 20,
        'frame_subsample': 12
    }
    
    # Load config from file if provided
    if args.data_config:
        with open(args.data_config, 'r') as f:
            config = json.load(f)
    else:
        config = default_config
        print("Using default configuration. Consider providing --data_config for custom settings.")
    
    # Initialize visualizer
    visualizer = TrajectoryVisualizer(
        model_checkpoint_path=args.model_path,
        config=config,
        device=args.device
    )
    
    # Load dataset
    classes = ['Pedestrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus']
    
    print("Loading dataset...")
    dataset = OptimizedMultiAgentSequenceDataset(
        drone_data_root=config['drone_data_root'],
        original_dataset_root=config['original_dataset_root'],
        classes=classes,
        T_past=config['T_past'],
        T_future=config['T_future'],
        use_deltas=True,
        normalize_positions=True,
        cache_dir=config['cache_dir'],
        lazy_loading=True,
        num_workers=config['num_workers'],
        max_agents=config['max_agents'],
        frame_subsample=config['frame_subsample']
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create visualizations
    visualizer.create_visualizations(
        dataset=dataset,
        num_sequences=args.num_sequences,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()