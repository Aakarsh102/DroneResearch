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

class DebugTrajectoryVisualizer:
    """Debug version of trajectory visualizer with detailed analysis"""
    
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
        
        # Enhanced color scheme for better visibility
        self.colors = {
            'past': '#1f77b4',           # Blue for past trajectory  
            'predicted': '#ff0000',      # Bright red for predicted trajectory
            'ground_truth': '#2ca02c',   # Green for ground truth future
            'observation': '#ff7f0e',    # Orange for observed positions
            'unobserved': '#aec7e8'      # Light blue for unobserved positions
        }
        
    def analyze_predictions(self, batch, predictions):
        """Analyze prediction quality and detect issues"""
        print("\n" + "="*50)
        print("PREDICTION ANALYSIS")
        print("="*50)
        
        batch_size = batch['past_positions'].size(0)
        
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}, dtype: {value.dtype}")
                print(f"  Range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                print(f"  Mean: {value.mean().item():.4f}, Std: {value.std().item():.4f}")
                
                # Check for NaN or inf
                if torch.isnan(value).any():
                    print(f"  ⚠️  WARNING: Contains NaN values!")
                if torch.isinf(value).any():
                    print(f"  ⚠️  WARNING: Contains infinite values!")
                    
                # Check if all predictions are identical (model not learning)
                if value.numel() > 1 and value.std().item() < 1e-6:
                    print(f"  ⚠️  WARNING: All predictions are nearly identical!")
        
        # Compare predictions vs ground truth
        if 'future_positions_mu' in predictions:
            pred_pos = predictions['future_positions_mu']
            true_pos = batch['future_positions_orig']
            temporal_mask = batch['temporal_masks_future']
            
            # Calculate errors only for valid positions
            valid_mask = temporal_mask > 0
            if valid_mask.sum() > 0:
                pred_valid = pred_pos[valid_mask]
                true_valid = true_pos[valid_mask]
                
                # Calculate displacement errors
                errors = torch.norm(pred_valid - true_valid, dim=-1)
                
                print(f"\nDisplacement Errors (valid positions only):")
                print(f"  Mean error: {errors.mean().item():.2f} pixels")
                print(f"  Std error: {errors.std().item():.2f} pixels")
                print(f"  Max error: {errors.max().item():.2f} pixels")
                print(f"  Min error: {errors.min().item():.2f} pixels")
                
                # Check if predictions are just copying last past position
                past_pos = batch['past_positions_orig']
                agent_mask = batch['agent_masks']
                
                for b in range(min(3, batch_size)):  # Check first 3 sequences
                    print(f"\nSequence {b}:")
                    valid_agents = torch.where(agent_mask[b] > 0)[0]
                    
                    for agent_idx in valid_agents[:3]:  # Check first 3 agents
                        # Get last past position
                        past_valid = batch['temporal_masks_past'][b, agent_idx] > 0
                        if past_valid.sum() > 0:
                            last_past = past_pos[b, agent_idx][past_valid][-1]
                            
                            # Get first predicted position
                            future_valid = temporal_mask[b, agent_idx] > 0
                            if future_valid.sum() > 0:
                                first_pred = pred_pos[b, agent_idx][future_valid][0]
                                first_true = batch['future_positions_orig'][b, agent_idx][future_valid][0]
                                
                                # Check distances
                                pred_dist = torch.norm(first_pred - last_past).item()
                                true_dist = torch.norm(first_true - last_past).item()
                                pred_error = torch.norm(first_pred - first_true).item()
                                
                                print(f"    Agent {agent_idx}: pred_move={pred_dist:.1f}px, "
                                      f"true_move={true_dist:.1f}px, error={pred_error:.1f}px")
                                
                                if pred_dist < 1.0:
                                    print(f"      ⚠️  Model barely moving from last position!")
        
        print("="*50 + "\n")
    
    def load_reference_image(self, location, video, original_dataset_root):
        """Load reference image for a video"""
        image_path = Path(original_dataset_root) / "annotations" / location / video / "reference.jpg"
        
        if not image_path.exists():
            print(f"Warning: Reference image not found at {image_path}")
            return None
            
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load image at {image_path}")
                return None
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
        
        denormalized = positions * std + mean
        return denormalized
    
    def visualize_sequence_debug(self, batch, predictions, idx, dataset, output_dir, sequence_name):
        """Enhanced visualization with debugging info"""
        location = batch['location'][idx] if isinstance(batch['location'], list) else batch['location'][idx].item()
        video = batch['video'][idx] if isinstance(batch['video'], list) else batch['video'][idx].item()
        
        print(f"\n--- Debugging Sequence {idx}: {location}/{video} ---")
        
        # Load reference image
        reference_img = self.load_reference_image(location, video, dataset.orig_root)
        
        # Extract positions and masks for this sequence
        past_pos = batch['past_positions_orig'][idx].cpu().numpy()
        future_pos_true = batch['future_positions_orig'][idx].cpu().numpy()
        future_pos_pred = predictions['future_positions_mu'][idx].cpu().numpy()
        
        obs_mask = batch['obs_masks'][idx].cpu().numpy()
        temporal_mask_past = batch['temporal_masks_past'][idx].cpu().numpy()
        temporal_mask_future = batch['temporal_masks_future'][idx].cpu().numpy()
        agent_mask = batch['agent_masks'][idx].cpu().numpy()
        agent_ids = batch['agent_ids'][idx].cpu().numpy()
        agent_labels = batch['agent_labels'][idx].cpu().numpy()
        
        print(f"Valid agents: {agent_mask.sum()}")
        
        # Denormalize predicted positions if needed
        if dataset.normalize_positions:
            future_pos_pred_reshaped = future_pos_pred.reshape(-1, 2)
            future_pos_pred_denorm = self.denormalize_positions(
                future_pos_pred_reshaped, location, video, dataset
            ).reshape(future_pos_pred.shape)
        else:
            future_pos_pred_denorm = future_pos_pred
        
        # Detailed analysis of each agent
        valid_agents = np.where(agent_mask > 0)[0]
        for agent_idx in valid_agents:
            agent_id = agent_ids[agent_idx]
            if agent_id == -1:
                continue
                
            print(f"\nAgent {agent_idx} (ID: {agent_id}):")
            
            # Past analysis
            past_valid = temporal_mask_past[agent_idx] > 0
            if past_valid.sum() > 0:
                past_traj = past_pos[agent_idx][past_valid]
                print(f"  Past: {past_valid.sum()} valid frames")
                print(f"    Range: X[{past_traj[:, 0].min():.1f}, {past_traj[:, 0].max():.1f}], "
                      f"Y[{past_traj[:, 1].min():.1f}, {past_traj[:, 1].max():.1f}]")
            
            # Future analysis
            future_valid = temporal_mask_future[agent_idx] > 0
            if future_valid.sum() > 0:
                future_true = future_pos_true[agent_idx][future_valid]
                future_pred = future_pos_pred_denorm[agent_idx][future_valid]
                
                print(f"  Future: {future_valid.sum()} valid frames")
                print(f"    True range: X[{future_true[:, 0].min():.1f}, {future_true[:, 0].max():.1f}], "
                      f"Y[{future_true[:, 1].min():.1f}, {future_true[:, 1].max():.1f}]")
                print(f"    Pred range: X[{future_pred[:, 0].min():.1f}, {future_pred[:, 0].max():.1f}], "
                      f"Y[{future_pred[:, 1].min():.1f}, {future_pred[:, 1].max():.1f}]")
                
                # Calculate error
                errors = np.linalg.norm(future_pred - future_true, axis=1)
                print(f"    Errors: mean={errors.mean():.1f}px, max={errors.max():.1f}px")
                
                # Check if prediction is just copying last past position
                if past_valid.sum() > 0:
                    last_past = past_pos[agent_idx][past_valid][-1]
                    first_pred = future_pred[0]
                    copy_dist = np.linalg.norm(first_pred - last_past)
                    print(f"    Distance from last past position: {copy_dist:.1f}px")
                    
                    if copy_dist < 5.0:
                        print(f"    ⚠️  WARNING: Prediction barely moves from last position!")
        
        # Create enhanced visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left plot: Normal view
        self._plot_trajectories(ax1, reference_img, past_pos, future_pos_true, future_pos_pred_denorm,
                               obs_mask, temporal_mask_past, temporal_mask_future, agent_mask, 
                               agent_ids, agent_labels, valid_agents, "Normal View")
        
        # Right plot: Zoomed view focusing on trajectories
        self._plot_trajectories(ax2, None, past_pos, future_pos_true, future_pos_pred_denorm,
                               obs_mask, temporal_mask_past, temporal_mask_future, agent_mask, 
                               agent_ids, agent_labels, valid_agents, "Trajectory Focus (No Background)")
        
        plt.suptitle(f'DEBUG: {location}/{video} - Frame Subsample: {batch["frame_subsample_rate"][idx]} | '
                    f'Valid Agents: {len(valid_agents)}', fontsize=16, fontweight='bold')
        
        # Save plot
        output_path = Path(output_dir) / f'debug_{sequence_name}.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved debug visualization: {output_path}")
        
        return {
            'location': location,
            'video': video,
            'num_agents': len(valid_agents),
            'agent_ids': agent_ids[valid_agents].tolist(),
            'agent_classes': [self.classes[agent_labels[i]] if 0 <= agent_labels[i] < len(self.classes) 
                            else "Unknown" for i in valid_agents]
        }
    
    def _plot_trajectories(self, ax, reference_img, past_pos, future_pos_true, future_pos_pred_denorm,
                          obs_mask, temporal_mask_past, temporal_mask_future, agent_mask, 
                          agent_ids, agent_labels, valid_agents, title):
        """Plot trajectories with enhanced visibility"""
        
        # Show reference image if available
        if reference_img is not None:
            ax.imshow(reference_img, alpha=0.6)
            img_height, img_width = reference_img.shape[:2]
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0)
        else:
            # Set limits based on trajectory data
            all_pos = []
            for agent_idx in valid_agents:
                if agent_mask[agent_idx] > 0:
                    past_valid = temporal_mask_past[agent_idx] > 0
                    future_valid = temporal_mask_future[agent_idx] > 0
                    
                    if past_valid.sum() > 0:
                        all_pos.extend(past_pos[agent_idx][past_valid])
                    if future_valid.sum() > 0:
                        all_pos.extend(future_pos_true[agent_idx][future_valid])
                        all_pos.extend(future_pos_pred_denorm[agent_idx][future_valid])
            
            if len(all_pos) > 0:
                all_pos = np.array(all_pos)
                padding = 50
                ax.set_xlim(all_pos[:, 0].min() - padding, all_pos[:, 0].max() + padding)
                ax.set_ylim(all_pos[:, 1].min() - padding, all_pos[:, 1].max() + padding)
        
        # Plot trajectories for each valid agent with enhanced visibility
        for agent_idx in valid_agents:
            agent_id = agent_ids[agent_idx]
            if agent_id == -1:
                continue
                
            # Get agent class
            class_idx = agent_labels[agent_idx]
            class_name = self.classes[class_idx] if 0 <= class_idx < len(self.classes) else "Unknown"
            
            # Past trajectory
            past_valid = temporal_mask_past[agent_idx] > 0
            if np.any(past_valid):
                past_traj = past_pos[agent_idx][past_valid]
                obs_valid = obs_mask[agent_idx][past_valid]
                
                # Plot past points
                for i in range(len(past_traj)):
                    color = self.colors['observation'] if obs_valid[i] > 0 else self.colors['unobserved']
                    ax.scatter(past_traj[i, 0], past_traj[i, 1], 
                             c=color, s=40, alpha=0.9, edgecolors='black', linewidth=1, zorder=5)
                
                # Connect past points with thick line
                if len(past_traj) > 1:
                    ax.plot(past_traj[:, 0], past_traj[:, 1], 
                           color=self.colors['past'], linewidth=4, alpha=0.8, linestyle='-', zorder=4)
            
            # Future trajectories
            future_valid = temporal_mask_future[agent_idx] > 0
            if np.any(future_valid):
                future_traj_true = future_pos_true[agent_idx][future_valid]
                future_traj_pred = future_pos_pred_denorm[agent_idx][future_valid]
                
                # Plot PREDICTED trajectory FIRST (so it's behind ground truth)
                ax.plot(future_traj_pred[:, 0], future_traj_pred[:, 1], 
                       color=self.colors['predicted'], linewidth=6, alpha=0.9, 
                       linestyle='--', zorder=2)
                
                # Plot predicted points as large squares
                ax.scatter(future_traj_pred[:, 0], future_traj_pred[:, 1], 
                         c=self.colors['predicted'], s=60, alpha=0.9, 
                         edgecolors='white', linewidth=2, marker='s', zorder=3)
                
                # Plot ground truth trajectory SECOND (on top)
                ax.plot(future_traj_true[:, 0], future_traj_true[:, 1], 
                       color=self.colors['ground_truth'], linewidth=4, alpha=0.8, 
                       linestyle='-', zorder=6)
                
                # Plot ground truth points as circles
                ax.scatter(future_traj_true[:, 0], future_traj_true[:, 1], 
                         c=self.colors['ground_truth'], s=40, alpha=0.8, 
                         edgecolors='black', linewidth=1, zorder=7)
                
                # Add error lines between predicted and true positions
                for i in range(len(future_traj_pred)):
                    ax.plot([future_traj_pred[i, 0], future_traj_true[i, 0]], 
                           [future_traj_pred[i, 1], future_traj_true[i, 1]], 
                           color='purple', linewidth=1, alpha=0.5, linestyle=':', zorder=1)
            
            # Connection lines from past to future
            if np.any(past_valid) and np.any(future_valid):
                last_past = past_pos[agent_idx][past_valid][-1]
                first_future_true = future_pos_true[agent_idx][future_valid][0]
                first_future_pred = future_pos_pred_denorm[agent_idx][future_valid][0]
                
                # Connection lines
                ax.plot([last_past[0], first_future_true[0]], 
                       [last_past[1], first_future_true[1]], 
                       color=self.colors['ground_truth'], linewidth=2, alpha=0.5, linestyle=':', zorder=3)
                
                ax.plot([last_past[0], first_future_pred[0]], 
                       [last_past[1], first_future_pred[1]], 
                       color=self.colors['predicted'], linewidth=2, alpha=0.5, linestyle=':', zorder=3)
            
            # Enhanced agent annotation
            if np.any(past_valid):
                last_pos = past_pos[agent_idx][past_valid][-1]
                ax.annotate(f'ID:{agent_id}\n{class_name}', 
                           xy=(last_pos[0], last_pos[1]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, ha='left', va='bottom', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black'))
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (pixels)', fontsize=10)
        ax.set_ylabel('Y Position (pixels)', fontsize=10)
        
        # Enhanced legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors['past'], linewidth=4, label='Past Trajectory'),
            plt.Line2D([0], [0], color=self.colors['ground_truth'], linewidth=4, label='Ground Truth Future'),
            plt.Line2D([0], [0], color=self.colors['predicted'], linewidth=6, linestyle='--', label='Predicted Future'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['observation'], 
                      markersize=8, label='Observed Position'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['predicted'], 
                      markersize=8, label='Predicted Position'),
            plt.Line2D([0], [0], color='purple', linewidth=2, linestyle=':', label='Prediction Error')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def debug_model_predictions(self, dataset, num_sequences=3, output_dir='debug_visualizations'):
        """Create debug visualizations with detailed analysis"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=num_sequences,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False
        )
        
        print(f"Creating debug visualizations for {num_sequences} sequences...")
        
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
        
        # Analyze predictions
        self.analyze_predictions(batch, predictions)
        
        # Create debug visualizations
        sequence_info = []
        actual_sequences = min(num_sequences, batch['past_positions'].size(0))
        
        for idx in range(actual_sequences):
            sequence_name = f'sequence_{idx+1:02d}'
            info = self.visualize_sequence_debug(
                batch, predictions, idx, dataset, output_dir, sequence_name
            )
            sequence_info.append(info)
        
        # Save detailed analysis
        analysis_path = output_dir / 'prediction_analysis.txt'
        with open(analysis_path, 'w') as f:
            f.write("PREDICTION ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            # Write model info
            f.write(f"Model loaded from checkpoint\n")
            f.write(f"Classes: {self.classes}\n")
            f.write(f"Locations: {self.locations}\n\n")
            
            # Write prediction statistics
            if 'future_positions_mu' in predictions:
                pred_pos = predictions['future_positions_mu']
                f.write(f"Prediction tensor shape: {pred_pos.shape}\n")
                f.write(f"Prediction range: [{pred_pos.min().item():.4f}, {pred_pos.max().item():.4f}]\n")
                f.write(f"Prediction std: {pred_pos.std().item():.4f}\n\n")
            
            # Write sequence info
            for i, info in enumerate(sequence_info):
                f.write(f"Sequence {i+1}: {info['location']}/{info['video']}\n")
                f.write(f"  Agents: {info['num_agents']}\n")
                f.write(f"  Classes: {', '.join(set(info['agent_classes']))}\n\n")
        
        print(f"\nDebug analysis completed!")
        print(f"Files saved in: {output_dir}")
        print(f"Analysis summary: {analysis_path}")
        
        return sequence_info


def main():
    parser = argparse.ArgumentParser(description='Debug trajectory predictions')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='debug_visualizations',
                       help='Output directory for debug visualizations')
    parser.add_argument('--num_sequences', type=int, default=3,
                       help='Number of sequences to debug')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'drone_data_root': "../../../square_stanford_data",
        'original_dataset_root': "../../../stanford_data/archive",
        'cache_dir': "new_cache_16",
        'batch_size': 8,
        'num_workers': 0,
        'max_agents': 128,
        'T_past': 10,
        'T_future': 20,
        'frame_subsample': 12
    }
    
    # Initialize debug visualizer
    debugger = DebugTrajectoryVisualizer(
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
    
    # Run debug analysis
    debugger.debug_model_predictions(
        dataset=dataset,
        num_sequences=args.num_sequences,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()