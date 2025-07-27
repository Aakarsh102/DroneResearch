import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import argparse
from matplotlib.colors import hsv_to_rgb
import matplotlib.patches as patches

def load_annotations(annotation_file, classes=None):
    """Load annotations using exact same format as dataset"""
    try:
        df = pd.read_csv(annotation_file, sep=' ', header=None,
                        names=['trackId', 'xmin', 'ymin', 'xmax', 'ymax',
                              'frame', 'lost', 'occluded', 'generated', 'label'])
        
        # Filter by classes if specified (same as dataset)
        if classes:
            df = df[df['label'].isin(classes)]
        
        # Calculate center points (same as dataset)
        df['x'] = (df['xmin'] + df['xmax']) * 0.5
        df['y'] = (df['ymin'] + df['ymax']) * 0.5
        
        return df
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return None

def build_multi_agent_sample_indices(df, T_past=5, T_future=5, frame_subsample=12, 
                                   min_frames_per_agent=2, obs_sets=None):
    """
    Build sample indices using EXACT same logic as dataset's _build_multi_agent_sample_indices_parallel
    """
    
    indices = []
    
    # Get all unique frames (same as dataset)
    all_frames = sorted(df['frame'].unique())
    window_size = T_past + T_future
    
    print(f"Total frames available: {len(all_frames)} (range: {min(all_frames)}-{max(all_frames)})")
    print(f"Window size: {window_size}, Frame subsample: {frame_subsample}")
    
    # NEW: Frame subsampling logic for multi-agent (EXACT copy from dataset)
    for frame_offset in range(frame_subsample):
        # Calculate how many subsampled windows we can fit
        max_subsampled_windows = (len(all_frames) - frame_offset) // frame_subsample
        
        for start_subsample_idx in range(max_subsampled_windows - window_size + 1):
            # Calculate the actual frame indices we'll use
            window_frame_indices = []
            window_frames = []
            
            for j in range(window_size):
                frame_idx = frame_offset + (start_subsample_idx + j) * frame_subsample
                if frame_idx < len(all_frames):
                    window_frame_indices.append(frame_idx)
                    window_frames.append(all_frames[frame_idx])
            
            if len(window_frames) < window_size:
                continue

            past_frames = window_frames[:T_past]
            future_frames = window_frames[T_past:]

            window_df = df[df['frame'].isin(window_frames)]

            if len(window_df) == 0:  # Check window_df, not window_frames
                continue

            # Find agents that meet minimum frame requirement (EXACT same logic)
            valid_agents = []
            agent_data = {}

            for tid, agent_df in window_df.groupby('trackId'):
                agent_frames = set(agent_df['frame'].values)
                past_frames_count = len([f for f in past_frames if f in agent_frames])
                future_frames_count = len([f for f in future_frames if f in agent_frames])

                if past_frames_count >= min_frames_per_agent:
                    # Check observation mask if obs_sets provided
                    if obs_sets:
                        obs_mask = np.array([1.0 if (tid, int(f)) in obs_sets else 0.0 
                                           for f in past_frames if f in agent_frames])
                        if len(obs_mask) > 0 and obs_mask.sum() >= 2:
                            valid_agents.append(tid)
                            agent_data[tid] = {
                                'past_frames': past_frames_count,
                                'future_frames': future_frames_count,
                                'obs_count': obs_mask.sum()
                            }
                    else:
                        # No observation requirement - just use all valid agents
                        valid_agents.append(tid)
                        agent_data[tid] = {
                            'past_frames': past_frames_count,
                            'future_frames': future_frames_count,
                            'obs_count': past_frames_count  # Assume all observed
                        }

            if len(valid_agents) == 0:
                continue
            
            # Create sample index (EXACT same format as dataset)
            sample_idx = {
                'start_frame': window_frames[0],
                'end_frame': window_frames[-1],
                'window_frames': window_frames,
                'frame_offset': frame_offset,
                'subsample_start_idx': start_subsample_idx,
                'valid_agents': valid_agents,
                'agent_data': agent_data,
                'num_agents': len(valid_agents)
            }

            indices.append(sample_idx)
    
    print(f"Found {len(indices)} valid multi-agent trajectory windows")
    return indices

def load_multi_agent_sample_exact(df, sample_idx, max_agents=20, pad_value=-999.0, 
                                T_past=5, T_future=5, obs_sets=None, classes=None):
    """
    Load multi-agent sample using EXACT same logic as dataset's _load_multi_agent_sample_on_demand
    """
    
    window_frames = sample_idx['window_frames']
    valid_agents = sample_idx['valid_agents']
    
    past_frames = window_frames[:T_past]
    future_frames = window_frames[T_past:]
    
    # Initialize arrays for all agents (EXACT same as dataset)
    num_agents = min(len(valid_agents), max_agents)
    
    # Shape: (max_agents, T_past, 2), (max_agents, T_future, 2)
    past_positions = np.full((max_agents, T_past, 2), pad_value, dtype=np.float32)
    future_positions = np.full((max_agents, T_future, 2), pad_value, dtype=np.float32)
    
    # Masks: 1 = valid/present, 0 = invalid/absent (EXACT same as dataset)
    obs_masks = np.zeros((max_agents, T_past), dtype=np.float32)
    temporal_masks_past = np.zeros((max_agents, T_past), dtype=np.float32)
    temporal_masks_future = np.zeros((max_agents, T_future), dtype=np.float32)
    occ_masks = np.zeros((max_agents, T_future), dtype=np.float32)
    agent_masks = np.zeros(max_agents, dtype=np.float32)
    
    # Labels and metadata (EXACT same as dataset)
    agent_labels = np.full(max_agents, -1, dtype=np.int32)
    agent_ids = np.full(max_agents, -1, dtype=np.int32)
    
    # Create class to index mapping if classes provided
    cls2idx = {c: i for i, c in enumerate(classes)} if classes else {}
    
    # Process each valid agent (EXACT same logic as dataset)
    for agent_idx, track_id in enumerate(valid_agents[:max_agents]):
        # Get track data for this agent
        track_df = df[df['trackId'] == track_id].sort_values('frame').reset_index(drop=True)
        
        # Mark this agent slot as used (not padded)
        agent_masks[agent_idx] = 1.0
        agent_ids[agent_idx] = track_id
        
        # Create frame to index mapping for this track
        frame_to_idx = {frame: idx for idx, frame in enumerate(track_df['frame'].values)}
        
        # Fill past positions (EXACT same logic)
        for t, frame in enumerate(past_frames):
            if frame in frame_to_idx:
                data_idx = frame_to_idx[frame]
                coords = np.array([track_df.iloc[data_idx]['x'], track_df.iloc[data_idx]['y']])
                
                # Always use ground truth coordinates
                past_positions[agent_idx, t] = coords
                
                # Mark temporal presence
                temporal_masks_past[agent_idx, t] = 1.0
                
                # Set observation mask (1 if observed, 0 if GT but not observed)
                if obs_sets:
                    obs_masks[agent_idx, t] = 1.0 if (track_id, int(frame)) in obs_sets else 0.0
                else:
                    obs_masks[agent_idx, t] = 1.0  # Assume all observed
                
                # Get label from first valid frame
                if agent_labels[agent_idx] == -1:
                    label = track_df.iloc[data_idx]['label']
                    if classes and label in cls2idx:
                        agent_labels[agent_idx] = cls2idx[label]
                    else:
                        agent_labels[agent_idx] = hash(label) % 100  # Simple label encoding
        
        # Fill future positions (EXACT same logic)
        for t, frame in enumerate(future_frames):
            if frame in frame_to_idx:
                data_idx = frame_to_idx[frame]
                coords = np.array([track_df.iloc[data_idx]['x'], track_df.iloc[data_idx]['y']])
                
                # Always use ground truth coordinates
                future_positions[agent_idx, t] = coords
                
                # Mark temporal presence
                temporal_masks_future[agent_idx, t] = 1.0
                
                # Set occlusion (only meaningful where agent is present)
                occ_masks[agent_idx, t] = track_df.iloc[data_idx]['occluded']
    
    # Return EXACT same sample format as dataset
    sample = {
        'past_positions': past_positions,
        'future_positions': future_positions,
        'obs_masks': obs_masks,
        'temporal_masks_past': temporal_masks_past,
        'temporal_masks_future': temporal_masks_future,
        'occ_masks': occ_masks,
        'agent_masks': agent_masks,
        'agent_labels': agent_labels,
        'agent_ids': agent_ids,
        'start_frame': sample_idx['start_frame'],
        'num_valid_agents': num_agents,
        'subsampled_frames': window_frames,
        'frame_subsample_rate': len(window_frames) // (T_past + T_future) if len(window_frames) > 1 else 1
    }
    
    return sample

def generate_colors(num_colors):
    """Generate distinct colors for agents"""
    colors = []
    for i in range(num_colors):
        hue = i / max(num_colors, 1)
        saturation = 0.8 + 0.2 * (i % 2)
        value = 0.7 + 0.3 * ((i + 1) % 2)
        rgb = hsv_to_rgb([hue, saturation, value])
        colors.append(rgb)
    return colors

def plot_multi_agent_trajectories_exact(annotation_file, reference_image_path, output_path=None,
                                       T_past=5, T_future=5, frame_subsample=12, 
                                       max_agents=20, pad_value=-999.0,
                                       max_samples=3, classes=None, obs_sets=None):
    """
    Plot multi-agent trajectories using EXACT same method as dataset
    """
    
    # Load annotations
    df = load_annotations(annotation_file, classes)
    if df is None:
        return
    
    # Load reference image
    try:
        img = Image.open(reference_image_path)
        img_array = np.array(img)
    except Exception as e:
        print(f"Error loading reference image: {e}")
        return
    
    # Build sample indices using exact dataset method
    sample_indices = build_multi_agent_sample_indices(
        df, T_past, T_future, frame_subsample, obs_sets=obs_sets)
    
    if len(sample_indices) == 0:
        print("No valid samples found!")
        return
    
    # Limit number of samples to plot
    sample_indices = sample_indices[:max_samples]
    
    # Create subplots
    fig, axes = plt.subplots(1, len(sample_indices), figsize=(8*len(sample_indices), 10))
    if len(sample_indices) == 1:
        axes = [axes]
    
    for sample_idx, sample_info in enumerate(sample_indices):
        ax = axes[sample_idx]
        ax.imshow(img_array)
        ax.set_xlim(0, img_array.shape[1])
        ax.set_ylim(img_array.shape[0], 0)
        
        # Load sample using exact dataset method
        sample = load_multi_agent_sample_exact(
            df, sample_info, max_agents, pad_value, T_past, T_future, obs_sets, classes)
        
        # Extract data from sample (same format as dataset output)
        past_positions = sample['past_positions']
        future_positions = sample['future_positions']
        temporal_masks_past = sample['temporal_masks_past']
        temporal_masks_future = sample['temporal_masks_future']
        agent_masks = sample['agent_masks']
        agent_ids = sample['agent_ids']
        obs_masks = sample['obs_masks']
        
        # Generate colors for valid agents
        valid_agent_indices = np.where(agent_masks > 0)[0]
        colors = generate_colors(len(valid_agent_indices))
        
        print(f"Sample {sample_idx + 1}: {len(valid_agent_indices)} agents")
        
        # Plot each agent's trajectory
        for i, agent_idx in enumerate(valid_agent_indices):
            color = colors[i]
            agent_id = agent_ids[agent_idx]
            
            # Get valid past positions (where temporal_mask = 1)
            past_valid_indices = np.where(temporal_masks_past[agent_idx] > 0)[0]
            if len(past_valid_indices) > 0:
                past_pos = past_positions[agent_idx][past_valid_indices]
                past_obs = obs_masks[agent_idx][past_valid_indices]
                
                # Plot past trajectory
                if len(past_pos) > 1:
                    ax.plot(past_pos[:, 0], past_pos[:, 1], 
                           color=color, linewidth=3, alpha=0.8, solid_capstyle='round')
                
                # Plot past points - different markers for observed vs unobserved
                observed_indices = past_obs > 0.5
                if np.any(observed_indices):
                    ax.scatter(past_pos[observed_indices, 0], past_pos[observed_indices, 1], 
                              c=[color], s=80, alpha=0.9, marker='o',
                              edgecolors='white', linewidth=2, zorder=5, label='Observed')
                
                unobserved_indices = past_obs <= 0.5
                if np.any(unobserved_indices):
                    ax.scatter(past_pos[unobserved_indices, 0], past_pos[unobserved_indices, 1], 
                              c=[color], s=80, alpha=0.6, marker='s',
                              edgecolors='white', linewidth=2, zorder=5, label='GT (unobserved)')
            
            # Get valid future positions (where temporal_mask = 1)
            future_valid_indices = np.where(temporal_masks_future[agent_idx] > 0)[0]
            if len(future_valid_indices) > 0:
                future_pos = future_positions[agent_idx][future_valid_indices]
                
                # Plot future trajectory (dashed)
                if len(future_pos) > 1:
                    ax.plot(future_pos[:, 0], future_pos[:, 1], 
                           color=color, linewidth=3, alpha=0.6, 
                           linestyle='--', solid_capstyle='round')
                
                # Plot future points (triangles)
                ax.scatter(future_pos[:, 0], future_pos[:, 1], 
                          c=[color], s=80, alpha=0.7, marker='^',
                          edgecolors='white', linewidth=2, zorder=5)
            
            # Connect past to future if both exist
            if len(past_valid_indices) > 0 and len(future_valid_indices) > 0:
                last_past = past_positions[agent_idx][past_valid_indices[-1]]
                first_future = future_positions[agent_idx][future_valid_indices[0]]
                ax.plot([last_past[0], first_future[0]], 
                       [last_past[1], first_future[1]], 
                       color=color, linewidth=2, alpha=0.4, linestyle=':')
            
            # Add agent ID label
            if len(past_valid_indices) > 0:
                label_pos = past_positions[agent_idx][past_valid_indices[-1]]
                ax.annotate(f'Agent {int(agent_id)}', 
                           (label_pos[0], label_pos[1]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=color, alpha=0.8))
        
        # Set title with exact same info as dataset
        window_frames = sample['subsampled_frames']
        ax.set_title(f'Multi-Agent Sample {sample_idx + 1}\n'
                    f'Frames: {window_frames[0]}-{window_frames[-1]} '
                    f'(subsample={frame_subsample})\n'
                    f'Valid Agents: {len(valid_agent_indices)}/{max_agents}', 
                    fontsize=11, fontweight='bold')
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add sample statistics (same as dataset would have)
        stats_text = (f'T_past: {T_past}, T_future: {T_future}\n'
                     f'Frame offset: {sample_info["frame_offset"]}\n'
                     f'Subsample start: {sample_info["subsample_start_idx"]}\n'
                     f'Padding value: {pad_value}')
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
               fontsize=8, verticalalignment='bottom', fontfamily='monospace')
    
    plt.suptitle(f'Multi-Agent Dataset Trajectories (Exact Method)\n'
                f'T_past={T_past}, T_future={T_future}, frame_subsample={frame_subsample}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Multi-agent trajectory plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes

def main():
    parser = argparse.ArgumentParser(description='Plot multi-agent trajectories using exact dataset method')
    parser.add_argument('annotation_file', help='Path to annotations.txt file')
    parser.add_argument('reference_image', help='Path to reference.jpg image')
    parser.add_argument('--output', '-o', help='Output path for saved image')
    parser.add_argument('--T-past', type=int, default=5, help='Number of past timesteps (default: 5)')
    parser.add_argument('--T-future', type=int, default=5, help='Number of future timesteps (default: 5)')
    parser.add_argument('--frame-subsample', type=int, default=12, help='Frame subsampling interval (default: 12)')
    parser.add_argument('--max-agents', type=int, default=20, help='Maximum agents per sample (default: 20)')
    parser.add_argument('--max-samples', type=int, default=3, help='Maximum samples to plot (default: 3)')
    parser.add_argument('--pad-value', type=float, default=-999.0, help='Padding value (default: -999.0)')
    parser.add_argument('--classes', nargs='+', help='Filter by specific classes')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    if not os.path.exists(args.reference_image):
        print(f"Error: Reference image not found: {args.reference_image}")
        return
    
    # Plot trajectories using exact dataset method
    plot_multi_agent_trajectories_exact(
        annotation_file=args.annotation_file,
        reference_image_path=args.reference_image,
        output_path=args.output,
        T_past=args.T_past,
        T_future=args.T_future,
        frame_subsample=args.frame_subsample,
        max_agents=args.max_agents,
        max_samples=args.max_samples,
        pad_value=args.pad_value,
        classes=args.classes
    )

# Example usage matching your dataset parameters exactly
def plot_deathcircle_exact():
    """Plot deathCircle/video1 using exact dataset parameters"""
    annotation_file = "/Users/aakarshrai/Desktop/stanford_data/archive/annotations/deathCircle/video0/annotations.txt"
    reference_image = "/Users/aakarshrai/Desktop/stanford_data/archive/annotations/deathCircle/video0/reference.jpg"
    output_path = "deathCircle_subsampled_trajectories.png"
    
    plot_multi_agent_trajectories_exact(
        annotation_file=annotation_file,
        reference_image_path=reference_image,
        output_path=output_path,
        T_past=10,    # Your dataset default
        T_future=10,  # Your dataset default  
        frame_subsample=12,  # Your dataset default
        max_agents=20,  # Your dataset default
        pad_value=-999.0,  # Your dataset default
        max_samples=3
    )

if __name__ == "__main__":
    # main()
    plot_deathcircle_exact()




#  python src/sdd/prototype2/trajectory_plotter.py /Users/aakarshrai/Desktop/stanford_data/archive/annotations/deathCircle/video1/annotations.txt /Users/aakarshrai/Desktop/stanford_data/archive/annotations/deathCircle/video1/reference.jpg -o /Users/aakarshrai/Desktop/doer.png
