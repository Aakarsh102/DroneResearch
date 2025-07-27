#!/usr/bin/env python3
"""
visualize_predictions.py

Visualize trajectory prediction (past, predicted, and ground truth) over the scene's reference image.
Handles delta predictions and dataset normalization properly.
"""
import os
import argparse
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# Import your model and data utilities
from model_an import LandscapeAwareTrajectoryPredictor
from training_loop_an import create_data_loaders  # adjust import if your training script is named differently


def load_model(checkpoint_path, num_classes, locations, d_model, num_layers, T_past, T_future, device):
    """
    Load the pretrained trajectory predictor.
    """
    model = LandscapeAwareTrajectoryPredictor(
        num_classes, locations,
        d_model=d_model, num_layers=num_layers,
        T_past=T_past, T_future=T_future
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    # fix the positional‐encoding buffer if it's in the old shape
    if 'temporal_encoding.pe' in state_dict:
        pe = state_dict['temporal_encoding.pe']
        # detect old ordering: first dim much larger than second
        if pe.ndim == 3 and pe.shape[0] > pe.shape[1]:
            # swap dims 0 and 1 → (1, max_len, d_model)
            state_dict['temporal_encoding.pe'] = pe.transpose(0, 1)

    # now load
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def deltas_to_positions(start_pos, deltas):
    """Convert delta predictions to absolute positions"""
    if isinstance(deltas, torch.Tensor):
        deltas = deltas.detach().cpu().numpy()
    if isinstance(start_pos, torch.Tensor):
        start_pos = start_pos.detach().cpu().numpy()
    
    positions = np.zeros_like(deltas)
    current_pos = start_pos.copy()
    
    for i in range(len(deltas)):
        current_pos = current_pos + deltas[i]
        positions[i] = current_pos.copy()
    
    return positions


def denormalize_positions(normalized_coords, dataset, sample):
    """Denormalize coordinates back to original scale"""
    video_key = f"{sample['location']}_{sample['video']}"
    video_stats = dataset.get_video_stats(video_key)
    
    if video_stats is None:
        return normalized_coords
    
    if isinstance(normalized_coords, torch.Tensor):
        normalized_coords = normalized_coords.detach().cpu().numpy()
    
    # Denormalize: original = normalized * std + mean
    return normalized_coords * video_stats['std'] + video_stats['mean']
def visualize_sample(data_root, dataset, sample, predictions, save_dir=None):
    """
    Overlay past/pred/future on the scene image, using sample['*_orig'] if available.
    Returns True if plotted, False if skipped due to no past or no GT.
    """
    # 1) unwrap Subset → real dataset (only needed if you ever call get_video_stats)
    real_dataset = getattr(dataset, 'dataset', dataset)

    # 2) load reference image
    loc, vid = sample['location'], sample['video']
    img_dir = os.path.join(data_root, 'annotations', loc, vid)
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png','jpg','jpeg'))]
    if not imgs:
        raise FileNotFoundError(f"No ref image in {img_dir}")
    img = cv2.imread(os.path.join(img_dir, imgs[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 3) get past & GT in original coords if available, else denormalize
    if 'past_positions_orig' in sample and 'future_positions_orig' in sample:
        past = sample['past_positions_orig']    # shape (T_past,2), absolute coords
        gt   = sample['future_positions_orig']  # shape (T_future,2)
    else:
        # you can still fetch stats via real_dataset.get_video_stats(f"{loc}_{vid}")
        past = denormalize_positions(sample['past_positions'], 
                                     real_dataset.get_video_stats(f"{loc}_{vid}"))
        gt   = denormalize_positions(sample['future_positions'], 
                                     real_dataset.get_video_stats(f"{loc}_{vid}"))

    # 4) build predicted positions in orig coords
    if 'future_deltas' in predictions:
        # start from the last past_norm (normalized) if you must, else from past[-1]
        start = past[-1]  
        pred = deltas_to_positions(start, predictions['future_deltas'].cpu().numpy())
    else:
        pred = denormalize_positions(predictions['future_positions'].cpu(),
                                     real_dataset.get_video_stats(f"{loc}_{vid}"))

    # 5) masks
    existence = sample.get('existence_mask', np.ones(len(gt)))
    visibility = sample.get('visibility_mask', np.ones(len(gt)))
    # only plot where points actually lie within the image
    def in_frame(pts):
        return (pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)

    past_vis = in_frame(past)
    pred_vis = in_frame(pred) & existence.astype(bool)
    gt_vis   = in_frame(gt)   & (existence*visibility).astype(bool)

    # skip samples without visible history or GT
    if not past_vis.any() or not gt_vis.any():
        return False

    # 6) clamp to frame
    def clamp(pts):
        x = np.clip(pts[:,0], 0, w-1)
        y = np.clip(pts[:,1], 0, h-1)
        return np.stack([x,y], axis=1)

    plt.figure(figsize=(12,8))
    plt.imshow(img)
    ax = plt.gca()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal', 'box')

    # 7) plotting
    p = clamp(past[past_vis]);    plt.plot(p[:,0], p[:,1], 'b.-', lw=2, ms=6, label='Past')
    r = clamp(pred[pred_vis]);    plt.plot(r[:,0], r[:,1], 'r.-', lw=2, ms=6, label='Pred')
    g = clamp(gt[gt_vis]);        plt.plot(g[:,0], g[:,1], 'g.-', lw=2, ms=6, label='GT')

    # occluded
    occ = in_frame(gt) & existence.astype(bool) & (~visibility.astype(bool))
    if occ.any():
        o = clamp(gt[occ]); plt.plot(o[:,0], o[:,1], 'yo', ms=6, alpha=0.7, label='GT (Occ)')

    # dashed connectors
    lp = clamp(np.array([past[past_vis][-1]]))[0]
    if pred_vis.any():
        fp = clamp(np.array([pred[pred_vis][0]]))[0]
        plt.plot([lp[0],fp[0]], [lp[1],fp[1]], 'r--', alpha=0.5, lw=1)
    if gt_vis.any():
        fg = clamp(np.array([gt[gt_vis][0]]))[0]
        plt.plot([lp[0],fg[0]], [lp[1],fg[1]], 'g--', alpha=0.5, lw=1)

    # endpoints & start
    st = clamp(np.array([past[past_vis][0]]))[0]; plt.plot(st[0], st[1], 'bo', ms=10, label='Start')
    if pred_vis.any():
        pe = clamp(np.array([pred[pred_vis][-1]]))[0]; plt.plot(pe[0], pe[1], 'rs', ms=10, label='Pred End')
    if gt_vis.any():
        ge = clamp(np.array([gt[gt_vis][-1]]))[0]; plt.plot(ge[0], ge[1], 'gs', ms=10, label='GT End')

    # title & legend
    tid  = sample.get('track_id','?')
    frm  = sample.get('start_frame','?')
    ival = sample.get('frame_interval','?')
    plt.title(f"{loc}/{vid} | Track {tid} | Frame {frm} | Δ {ival}")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.axis('off')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{loc}_{vid}_track{tid}_vis.png")
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print("Saved ➔", out)

    plt.show()
    return True



# def visualize_sample(data_root, dataset, sample, predictions, save_dir=None):
#     """
#     Overlay past trajectory, predicted trajectory, and ground truth onto the reference image for one sample.
#     """
#     loc = sample['location']
#     vid = sample['video']
#     img_dir = os.path.join(data_root, 'annotations', loc, vid)
    
#     # find first image in the folder
#     imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     if not imgs:
#         raise FileNotFoundError(f"No reference image found in {img_dir}")
#     img_path = os.path.join(img_dir, imgs[0])

#     # load and convert to RGB
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Extract trajectories from sample (these are normalized if normalize_positions=True)
#     past_positions_norm = sample['past_positions']  # (T_past, 2)
#     future_positions_gt_norm = sample['future_positions']  # (T_future, 2)
    
#     # Get existence and visibility masks
#     existence_mask = sample.get('existence_mask', np.ones(len(future_positions_gt_norm)))
#     visibility_mask = sample.get('visibility_mask', np.ones(len(future_positions_gt_norm)))
    
#     # Denormalize positions to original coordinates
#     past_positions = denormalize_positions(past_positions_norm, dataset, sample)
#     future_positions_gt = denormalize_positions(future_positions_gt_norm, dataset, sample)
    
#     # Handle model predictions
#     if 'future_deltas' in predictions:
#         # Model predicts deltas
#         future_deltas = predictions['future_deltas'].detach().cpu().numpy()  # (T_future, 2)
        
#         # Convert deltas to positions (starting from last past position)
#         start_pos = past_positions_norm[-1]  # Use normalized start position
#         predicted_positions_norm = deltas_to_positions(start_pos, future_deltas)
        
#         # Denormalize predicted positions
#         predicted_positions = denormalize_positions(predicted_positions_norm, dataset, sample)
        
#     elif 'future_positions' in predictions:
#         # Model predicts absolute positions (normalized)
#         predicted_positions_norm = predictions['future_positions'].detach().cpu().numpy()
#         predicted_positions = denormalize_positions(predicted_positions_norm, dataset, sample)
        
#     else:
#         raise KeyError(f"Expected 'future_deltas' or 'future_positions' in predictions. Found: {list(predictions.keys())}")

#     plt.figure(figsize=(12, 8))
#     plt.imshow(img)
    
#     # Plot past trajectory (blue)
#     if len(past_positions) > 0:
#         plt.plot(past_positions[:, 0], past_positions[:, 1], 'b.-', linewidth=2, markersize=6, label='Past Trajectory')
    
#     # Plot predicted trajectory (red) - only for existing frames
#     valid_pred_mask = existence_mask.astype(bool)
#     if np.any(valid_pred_mask):
#         valid_predicted = predicted_positions[valid_pred_mask]
#         plt.plot(valid_predicted[:, 0], valid_predicted[:, 1], 'r.-', linewidth=2, markersize=6, label='Predicted Trajectory')
    
#     # Plot ground truth future trajectory (green) - only for existing and visible frames
#     valid_gt_mask = (existence_mask * visibility_mask).astype(bool)
#     if np.any(valid_gt_mask):
#         valid_gt = future_positions_gt[valid_gt_mask]
#         plt.plot(valid_gt[:, 0], valid_gt[:, 1], 'g.-', linewidth=2, markersize=6, label='Ground Truth')
    
#     # Plot occluded/invisible ground truth points (yellow dots)
#     occluded_mask = existence_mask.astype(bool) & (~visibility_mask.astype(bool))
#     if np.any(occluded_mask):
#         occluded_gt = future_positions_gt[occluded_mask]
#         plt.plot(occluded_gt[:, 0], occluded_gt[:, 1], 'yo', markersize=4, alpha=0.7, label='GT (Occluded)')
    
#     # Add connection lines between past and future (dashed)
#     if len(past_positions) > 0:
#         last_past = past_positions[-1]
        
#         # Connect to predicted trajectory
#         if np.any(valid_pred_mask):
#             first_pred = predicted_positions[np.where(valid_pred_mask)[0][0]]
#             plt.plot([last_past[0], first_pred[0]], [last_past[1], first_pred[1]], 'r--', alpha=0.5, linewidth=1)
        
#         # Connect to ground truth trajectory
#         if np.any(valid_gt_mask):
#             first_gt = future_positions_gt[np.where(valid_gt_mask)[0][0]]
#             plt.plot([last_past[0], first_gt[0]], [last_past[1], first_gt[1]], 'g--', alpha=0.5, linewidth=1)

#     # Mark important points
#     if len(past_positions) > 0:
#         plt.plot(past_positions[0, 0], past_positions[0, 1], 'bo', markersize=10, label='Start')
    
#     # Mark end points (only if they exist)
#     if np.any(valid_pred_mask):
#         last_pred_idx = np.where(valid_pred_mask)[0][-1]
#         plt.plot(predicted_positions[last_pred_idx, 0], predicted_positions[last_pred_idx, 1], 
#                 'rs', markersize=10, label='Predicted End')
    
#     if np.any(valid_gt_mask):
#         last_gt_idx = np.where(valid_gt_mask)[0][-1]
#         plt.plot(future_positions_gt[last_gt_idx, 0], future_positions_gt[last_gt_idx, 1], 
#                 'gs', markersize=10, label='GT End')

#     # Add info about track and frame
#     track_id = sample.get('track_id', 'unknown')
#     start_frame = sample.get('start_frame', 'unknown')
#     frame_interval = sample.get('frame_interval', 'unknown')
    
#     plt.title(f"Trajectory Prediction for {loc}/{vid}\nTrack ID: {track_id}, Start Frame: {start_frame}, Interval: {frame_interval}")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.axis('off')  # remove axes for cleaner visualization
    
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         out_file = os.path.join(save_dir, f"{loc}_{vid}_track{track_id}_trajectory_vis.png")
#         plt.savefig(out_file, dpi=200, bbox_inches='tight')
#         print(f"Saved visualization to {out_file}")
#     plt.show()


class Arguments():
    def __init__(self, data_root, drone_root, checkpoint, locations, classes, d_model, num_layers, T_past, T_future, batch_size, num_samples, save_dir):
        self.data_root = data_root
        self.drone_root = drone_root
        self.checkpoint = checkpoint
        self.locations = locations
        self.classes = classes
        self.d_model = d_model
        self.num_layers = num_layers
        self.T_past = T_past
        self.T_future = T_future
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.save_dir = save_dir


def main():
    args = Arguments(data_root="/Users/aakarshrai/Desktop/stanford_data/archive",
                     drone_root="/Users/aakarshrai/Desktop/square_stanford_data",
                     checkpoint="src/sdd/prototype1/models/Landscape1.pth",
                     locations=["deathCircle", "bookstore", "coupa", "gates", "hyang", "little", "nexus", "quad"],
                     classes=['Pedestrian','Biker','Skater','Cart','Car','Bus'],
                     d_model=64,
                     num_layers=1,
                     T_past=10,
                     T_future=20,
                     batch_size=1,
                     num_samples=5,
                     save_dir="visualizations")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        args.checkpoint,
        num_classes=len(args.classes),
        locations=args.locations,
        d_model=args.d_model,
        num_layers=args.num_layers,
        T_past=args.T_past,
        T_future=args.T_future,
        device=device,
    )

    # Get the data loaders with the same parameters as training
    _, val_loader, _ = create_data_loaders(
        drone_data_root=args.drone_root,
        original_dataroot=args.data_root,
        classes=args.classes,
        T_past=args.T_past,
        T_future=args.T_future,
        use_deltas=True,  # Important: match your training settings
        normalize_positions=True,  # Important: match your training settings  
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.2,
        num_workers=2
    )
    
    # Get access to the dataset for denormalization
    dataset = val_loader.dataset

    count = 0
    for batch in val_loader:
        if count >= args.num_samples:
            break
            
        # move tensors to device
        batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        with torch.no_grad():
            preds = model(batch_dev)

        # extract single sample (first item in batch)
        sample = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:  # scalar
                    sample[k] = v[0].item()
                else:
                    sample[k] = v[0].cpu().numpy()
            else:
                sample[k] = v[0] if isinstance(v, (list, tuple)) else v
        
        # extract predictions for first sample
        predictions = {k: v[0] for k, v in preds.items()}
        
        try:
            visualize_sample(args.data_root, dataset, sample, predictions, save_dir=args.save_dir)
            count += 1
        except Exception as e:
            print(f"Error visualizing sample {count}: {e}")
            print("Available prediction keys:", list(predictions.keys()))
            print("Available sample keys:", list(sample.keys()))
            import traceback
            traceback.print_exc()
            count += 1


if __name__ == '__main__':
    main()