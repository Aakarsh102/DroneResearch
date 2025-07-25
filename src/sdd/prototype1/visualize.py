#!/usr/bin/env python3
"""
visualize_predictions.py

Visualize trajectory prediction mean and variance over the scene's reference image.
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
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


def visualize_sample(data_root, sample, predictions, save_dir=None):
    """
    Overlay predicted mean and variance onto the reference image for one sample.
    """
    loc = sample['location']
    vid = sample['video']
    img_dir = os.path.join(data_root, 'annotations', loc, vid)
    # find first image in the folder
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imgs:
        raise FileNotFoundError(f"No reference image found in {img_dir}")
    img_path = os.path.join(img_dir, imgs[0])

    # load and convert to RGB
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # extract predictions
    mu = predictions['future_positions_mu'].detach().cpu().numpy()   # shape: (T_future, 2)
    logvar = predictions['future_positions_logvar'].detach().cpu().numpy()  # (T_future, 2)
    var = np.exp(logvar)
    std = np.sqrt(var)

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    # plot trajectory mean
    plt.plot(mu[:, 0], mu[:, 1], 'r.-', label='Predicted Mean')
    # plot uncertainty as translucent circles
    ax = plt.gca()
    for i in range(mu.shape[0]):
        radius = std[i].mean()  # average uncertainty radius
        circ = plt.Circle((mu[i, 0], mu[i, 1]), radius, color='r', alpha=0.2)
        ax.add_patch(circ)

    plt.title(f"Prediction for {loc}/{vid}")
    plt.legend()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, f"{loc}_{vid}_vis.png")
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        print(f"Saved visualization to {out_file}")
    plt.show()

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
    # parser = argparse.ArgumentParser(description="Visualize trajectory predictions on reference images")
    # parser.add_argument('--data_root',   required=True, help="Path to original_dataset_root")
    # parser.add_argument('--drone_root',  required=True, help="Path to drone_data_root")
    # parser.add_argument('--checkpoint',  required=True, help="Model checkpoint (.pth) to load")
    # parser.add_argument('--locations',   nargs='+', default=["bookstore","coupa","deathCircle","gates","hyang","little","nexus","quad"],
    #                     help="List of location names")
    # parser.add_argument('--classes',     nargs='+', default=['Pedestrian','Biker','Skater','Cart','Car','Bus'])
    # parser.add_argument('--d_model',     type=int, default=64)
    # parser.add_argument('--num_layers',  type=int, default=1)
    # parser.add_argument('--T_past',      type=int, default=10)
    # parser.add_argument('--T_future',    type=int, default=20)
    # parser.add_argument('--batch_size',  type=int, default=1)
    # parser.add_argument('--num_samples', type=int, default=5,
    #                     help="How many samples from the validation set to visualize")
    # parser.add_argument('--save_dir',    default=None, help="Directory to save overlay images")
    args = Arguments(data_root = "/Users/aakarshrai/Desktop/stanford_data/archive",
                     drone_root = "/Users/aakarshrai/Desktop/square_stanford_data",
                     checkpoint = "src/sdd/prototype1/models/Landscape1.pth",
                     locations = ["deathCircle", "bookstore", "coupa", "gates", "hyang", "little", "nexus", "quad"],
                     classes = ['Pedestrian','Biker','Skater','Cart','Car','Bus'],
                     d_model = 64,
                     num_layers = 1,
                     T_past = 10,
                     T_future = 20,
                     batch_size = 1,
                     num_samples = 5,
                     save_dir = "visualizations")

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

    # We only need the validation loader for visualization
    _, val_loader, _ = create_data_loaders(
        drone_data_root=args.drone_root,
        original_dataroot=args.data_root,
        classes=args.classes,
        T_past=args.T_past,
        T_future=args.T_future,
        use_deltas=True,
        normalize_positions=False,
        batch_size=args.batch_size,
        train_split=0.0,
        val_split=1.0,
        num_workers=0
    )

    count = 0
    for batch in val_loader:
        if count >= args.num_samples:
            break
        # move tensors to device
        batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            preds = model(batch_dev)

        # extract single sample
        sample = {k: (v[0].item() if isinstance(v, torch.Tensor) and v.numel()==1 else
                     (v[0].cpu().numpy() if isinstance(v, torch.Tensor) else v[0]))
                  for k, v in batch.items()}
        predictions = {k: v[0] for k, v in preds.items()}
        visualize_sample(args.data_root, sample, predictions, save_dir=args.save_dir)
        count += 1

if __name__ == '__main__':
    main()
