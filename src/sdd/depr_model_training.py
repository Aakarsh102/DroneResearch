import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class StanfordMultiDroneDataset(Dataset):
    """
    PyTorch Dataset for multi-drone Stanford data.

    For each sequence, returns:
      - input: Tensor[(1+K)*T_past, H, W] with static terrain + K class masks over T_past frames
      - target_pos: Tensor[N_max, T_future, 2] future (x,y) positions per agent
      - target_mask: Tensor[N_max]  (1 if real agent, 0 if pad)
    """
    def __init__(self,
                 drone_data_root: str,
                 original_dataset_root: str,
                 classes: list,
                 T_past: int = 20,
                 T_future: int = 10,
                 blur_sigma: float = 2.0,
                 N_max: int = 16):
        self.drone_data_root = drone_data_root
        self.orig_root = original_dataset_root
        self.classes = classes         # e.g. ['Pedestrian','Biker',...]
        self.cls2idx = {c:i for i,c in enumerate(self.classes)}
        self.K = len(classes)
        self.T_past = T_past
        self.T_future = T_future
        self.blur_sigma = blur_sigma
        self.N_max = N_max

        # scan all (location, video) folders and build index of sequences
        self.index = []  # each entry = (location, video, start_frame)
        for loc in os.listdir(self.drone_data_root):
            loc_path = os.path.join(self.drone_data_root, loc)
            if not os.path.isdir(loc_path):
                continue
            # assume drone1 holds full frame range
            d1 = os.path.join(loc_path, 'drone1')
            # load annotations per video
            for fn in os.listdir(d1):
                if not fn.endswith('_annotations.txt'): continue
                vid = fn.replace('_annotations.txt','')
                # load full global annotations via combining drones
                # for simplicity, assume contiguous frames from min to max
                # here we find min/max by scanning drone1 only
                ann = np.loadtxt(os.path.join(d1, fn),
                                 dtype={'names':('track','xmin','ymin','xmax','ymax','frame','l','o','g','lbl'),
                                        'formats':('i4','f4','f4','f4','f4','i4','i4','i4','i4','U10')},
                                 ndmin=1)
                frames = ann['frame']
                if len(frames)==0: continue
                min_f, max_f = frames.min(), frames.max()
                # register all valid start frames
                for t0 in range(min_f, max_f - (self.T_past + self.T_future) + 2):
                    self.index.append((loc, vid, int(t0)))

        # load reference sizes from one example
        sample_loc, sample_vid, _ = self.index[0]
        ref = cv2.imread(os.path.join(
            self.orig_root, 'annotations', sample_loc, sample_vid, 'reference.jpg'),
            cv2.IMREAD_GRAYSCALE)
        self.H, self.W = ref.shape

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        loc, vid, t0 = self.index[idx]
        # load static terrain
        ref_img = cv2.imread(os.path.join(
            self.orig_root, 'maps', loc, vid, 'reference.jpg'),
            cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # dynamic input: build per-frame class masks
        dyn_frames = []
        for dt in range(self.T_past):
            t = t0 + dt
            # initialize masks: K channels
            mask = np.zeros((self.K, self.H, self.W), dtype=np.float32)
            # gather annotations from all drones
            for dr in os.listdir(os.path.join(self.drone_data_root, loc)):
                ann_file = os.path.join(self.drone_data_root, loc, dr, f"{vid}_annotations.txt")
                if not os.path.isfile(ann_file): continue
                data = np.loadtxt(ann_file, dtype=str)
                # fields: trackId, xmin, ymin, xmax, ymax, frame, lost, occl, gen, label
                for row in data:
                    frm = int(row[5])
                    if frm != t: continue
                    lbl = row[9].strip('"')
                    if lbl not in self.cls2idx: continue
                    c = self.cls2idx[lbl]
                    xmin, ymin = int(float(row[1])), int(float(row[2]))
                    xmax, ymax = int(float(row[3])), int(float(row[4]))
                    mask[c, ymin:ymax, xmin:xmax] = 1.0
            # optional blur
            for c in range(self.K):
                mask[c] = cv2.GaussianBlur(mask[c], ksize=(0,0), sigmaX=self.blur_sigma)
            dyn_frames.append(mask)

        # stack: first channel = terrain, then per-frame class masks
        # result shape: ( (1 + K)*T_past, H, W )
        inp = []
        for t in range(self.T_past):
            inp.append(ref_img[None, ...])           # 1×H×W
            inp.append(dyn_frames[t])                # K×H×W
        inp = np.concatenate(inp, axis=0)
        inp = torch.from_numpy(inp)

        # build targets: find agents present at t_end
        agents = {}
        t_end = t0 + self.T_past - 1
        for dr in os.listdir(os.path.join(self.drone_data_root, loc)):
            ann_file = os.path.join(self.drone_data_root, loc, dr, f"{vid}_annotations.txt")
            if not os.path.isfile(ann_file): continue
            data = np.loadtxt(ann_file, dtype=str)
            for row in data:
                if int(row[5]) != t_end: continue
                track = int(row[0])
                # center
                x = (float(row[1]) + float(row[3]))/2.0
                y = (float(row[2]) + float(row[4]))/2.0
                lbl = row[9].strip('"')
                agents[track] = {'label': lbl, 'xy0':(x,y)}
        # now for each agent get future T_future positions
        N = len(agents)
        pos = np.zeros((self.N_max, self.T_future, 2), dtype=np.float32)
        msk = np.zeros((self.N_max,), dtype=np.float32)
        for i,(track, info) in enumerate(agents.items()):
            if i>=self.N_max: break
            msk[i] = 1.0
            for f in range(self.T_future):
                tf = t_end + 1 + f
                # search annotation for this track at tf
                found=False
                for dr in os.listdir(os.path.join(self.drone_data_root, loc)):
                    ann_file = os.path.join(self.drone_data_root, loc, dr, f"{vid}_annotations.txt")
                    data = np.loadtxt(ann_file, dtype=str)
                    for row in data:
                        if int(row[0])==track and int(row[5])==tf:
                            x = (float(row[1])+float(row[3]))/2.0
                            y = (float(row[2])+float(row[4]))/2.0
                            pos[i,f,0], pos[i,f,1] = x,y
                            found=True; break
                    if found: break
        target_pos = torch.from_numpy(pos)
        target_mask = torch.from_numpy(msk)

        return {
            'input': inp,                 # FloatTensor[(1+K)*T_past, H, W]
            'target_pos': target_pos,     # FloatTensor[N_max, T_future, 2]
            'target_mask': target_mask    # FloatTensor[N_max]
        }
    


class TrajectoryViT(nn.Module):
    def __init__(self, image_size:int, patch_size:int, 
                 in_channels:int, d_model:int, num_layers:int,
                 num_heads:int, N_max:int, T_future:int):
        super(TrajectoryViT, self).__init__()
        self.N_max = N_max
        self.T_future = T_future
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embed == nn.Linear(self.patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = num_heads,
                                                        dim_feedforward = d_model * 4, dropout = 0.1,
                                                        activation = 'gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = self.num_layers)

        # MLP head produces the mean and log-variances for the future trajectory
        # Output shape: (B, N_max * T_future * 4), 4 cuz (mux, muy, logvarx, logvary)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.N_max * self.T_future * 4)
        )

        
    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.unfold(2, P, P).unfold(3, P, P) # (B, C, H/P, W/P, P, P)
        x = x.contiguous().view(B, C, -1, P, P) # (B, C, N, P, P)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, self.num_patches, -1)  # (B, N, patch_dim)
        tokens = self.patch_embed(x) # (B, N, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens += self.pos_embed # (B, N+1, d_model)
        encoded = self.transformer(tokens)  # (B, N+1, d_model)
        params = self.mlp_head(encoded[:, 0])  # Use cls token output # (B, d_model)
        # params = (B, N_max * T_future * 4)
        params = params.view(B, self.N_max, self.T_future, 4)
        # params shape: (B, N_max, T_future, 4)
        mu = params[..., :2]
        logvar = params[..., 2:]
        return mu, logvar
    
def gaussian_nll_loss(mu, logvar, target, mask):
    """
    Compute negative log-likelihood of diagonal Gaussian.
    mu, logvar: (B, N_max, T_future, 2)
    target:     (B, N_max, T_future, 2)
    mask:       (B, N_max)
    
    """
    var = torch.ex(logvar)
    mask_exp = mask.unsqueeze(-1).unsqueeze(-1)  # (B, N_max, 1, 1)
    # Compute NLL: 0.5*(logvar + (x-mu)^2/var)
    nll = 0.5 * (logvar + ((target - mu) ** 2) / var)
    nll = nll.sum(dim = -1)  # Sum over (x, y) dimensions
    nll = nll * mask_exp  # Apply mask
    total_valid = mask_exp.sum() * mu.shape[2]
    return nll.sum() / total_valid.clamp_min(1)

def train_model(
    model: nn.Module,
    dataset:Dataset,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4
    ):
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            inp = batch['input'].to(device)           # (B, C, H, W)
            target = batch['target_pos'].to(device)   # (B, N_max, T_future, 2)
            mask = batch['target_mask'].to(device)    # (B, N_max)

            mu, logvar = model(inp)
            loss = gaussian_nll_loss(mu, logvar, target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  * inp.size(0)
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")



if __name__ == "__main__":
    dataset = StanfordMultiDroneDataset(drone_data_root= "square_stanford_data", 
                                        original_daaset_root = "stanford_data/archive",
                                        classes = ['Pedstrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus'],
                                        T_past = 10, T_future = 10, N_max = 20)
                                        
    # model = TrajectoryViT(image_size = )



 


# def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
#     """
#     Train the model on the provided dataset.

#     Args:
#         model: The neural network model to train.
#         dataloader: DataLoader providing training data.
#         criterion: Loss function.
#         optimizer: Optimizer for updating model weights.
#         num_epochs: Number of epochs to train.

#     Returns:
#         model: The trained model.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     for epoch in range(num_epochs):
#         model.train()  # Set model to training mode
#         running_loss = 0.0

#         for batch in dataloader:
#             inputs = batch['input'].to(device)
#             target_pos = batch['target_pos'].to(device)
#             target_mask = batch['target_mask'].to(device)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward pass

