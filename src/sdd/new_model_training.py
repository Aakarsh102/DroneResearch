import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AgentSequenceDataset(Dataset):
    """
    PyTorch Dataset yielding per-agent trajectory sequences with:
      - past coords in global frame
      - observation mask (which past frames were seen by any drone)
      - future coords (ground truth)
      - occlusion mask for future (which future frames are not occluded)
      - label (class index)
    """
    def __init__(self, drone_data_root, original_dataset_root, classes,
                 T_past=10, T_future=10):
        self.drone_data_root = drone_data_root
        self.orig_root = original_dataset_root
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        self.T_past = T_past
        self.T_future = T_future
        self.samples = []

        # Precompute obs sets per (loc,vid): which (trackId,frame) were seen
        obs = {}  # obs[(loc,vid)] = set of (trackId,frame)
        for loc in os.listdir(drone_data_root):
            loc_path = os.path.join(drone_data_root, loc)
            if not os.path.isdir(loc_path): continue
            if loc == 'hyang': continue
            for dr in os.listdir(loc_path):
                dr_path = os.path.join(loc_path, dr)
                if not os.path.isdir(dr_path): continue
                # metadata gives nothing for obs
                for fname in os.listdir(dr_path):
                    if not fname.endswith("_annotations.txt"): continue
                    vid = fname.replace("_annotations.txt","")
                    key = (loc, vid)
                    obs.setdefault(key, set())
                    df = pd.read_csv(os.path.join(dr_path,fname), sep=' ', header=None,
                                     names=['trackId','xmin','ymin','xmax','ymax',
                                            'frame','lost','occluded','generated','label'])
                    for tid,fr in zip(df['trackId'], df['frame']):
                        obs[key].add((tid, int(fr)))

        # Build samples using original dataset for full tracks
        for loc in os.listdir(original_dataset_root + "/annotations"):
            if loc == 'hyang' or loc == ".DS_Store": continue
            if not os.path.isdir(loc_path):            # ← skip files like .DS_Store
                continue
            for vid in os.listdir(os.path.join(original_dataset_root,"annotations",loc)):
                # load original annotations
                orig_file = os.path.join(original_dataset_root,"annotations",loc,vid,"annotations.txt")
                if not os.path.isfile(orig_file): continue
                odf = pd.read_csv(orig_file, sep=' ', header=None,
                                  names=['trackId','xmin','ymin','xmax','ymax',
                                         'frame','lost','occluded','generated','label'])
                # compute centers
                odf['x'] = (odf.xmin + odf.xmax)/2
                odf['y'] = (odf.ymin + odf.ymax)/2
                # keep only labels in classes
                odf = odf[odf['label'].isin(self.classes)]
                # group by track
                key = (loc, vid)
                seen = obs.get(key, set())
                for tid, grp in odf.groupby('trackId'):
                    grp = grp.sort_values('frame')
                    frames = grp['frame'].values
                    coords = np.stack([grp['x'].values, grp['y'].values], axis=1)
                    occs   = (grp['occluded']==0).astype(np.float32).values
                    # build observation mask
                    obs_mask = np.array([1.0 if (tid, int(f)) in seen else 0.0 for f in frames],
                                        dtype=np.float32)
                    L = len(frames)
                    window = self.T_past + self.T_future
                    for i in range(L - window + 1):
                        # ensure continuous frames in original
                        if frames[i+self.T_past-1] - frames[i] != self.T_past-1:
                            continue
                        if frames[i+window-1]   - frames[i+self.T_past] != self.T_future-1:
                            continue
                        # require at least one observed in past
                        if obs_mask[i:i+self.T_past].sum() < 1:
                            continue
                        past   = coords[i:i+self.T_past]
                        future = coords[i+self.T_past:i+window]
                        past_obs = obs_mask[i:i+self.T_past]
                        future_occ = occs[i+self.T_past:i+window]
                        label = grp['label'].values[i+self.T_past-1]
                        idx = self.cls2idx[label]
                        self.samples.append({
                            'past': past.astype(np.float32),
                            'obs_mask': past_obs,
                            'future': future.astype(np.float32),
                            'occ_mask': future_occ.astype(np.float32),
                            'label': idx
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'past':     torch.from_numpy(s['past']),     # (T_past,2)
            'obs_mask': torch.from_numpy(s['obs_mask']), # (T_past,)
            'future':   torch.from_numpy(s['future']),   # (T_future,2)
            'occ_mask': torch.from_numpy(s['occ_mask']), # (T_future,)
            'label':    torch.tensor(s['label'])
        }

num_classes = 6
class TrajectoryModel(nn.Module):
    """
    Transformer encoder + GRU-based autoregressive decoder predicting
    future Gaussian (mu, logvar) per timestep, with mask embedding.
    """
    def __init__(self, T_past, T_future, labels, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.T_past = T_past
        self.T_future = T_future
        self.d_model = d_model
        self.labels = labels

        # input projection for coords
        self.input_proj = nn.Linear(2, d_model)
        # mask embeddings
        self.e_obs  = nn.Parameter(torch.randn(d_model))
        self.e_miss = nn.Parameter(torch.randn(d_model))
        # positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(T_past, d_model))
        self.label_emb = nn.Embedding(num_classes, d_model)

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # decoder GRU cell
        self.gru = nn.GRUCell(d_model, d_model)
        # output head: d_model -> 4 dims
        self.out_head = nn.Linear(d_model, 4)

    def forward(self, past, obs_mask, future=None):
        # past: (B, T_past, 2), obs_mask: (B, T_past)
        B = past.size(0)
        # embed coords
        x = self.input_proj(past)  # (B,T_past,d_model)
        # add mask embedding
        mask_embed = obs_mask.unsqueeze(-1)*(self.e_obs) + (1-obs_mask).unsqueeze(-1)*(self.e_miss)
        x = x + mask_embed
        # add pos emb
        x = x + self.pos_emb.unsqueeze(0)

        # label encoding 
        lbl = self.label_emb(self.labels.to(x.device)) 
        x = x + lbl


        # encoder
        x_t = x.permute(1,0,2)     # (T_past,B,d_model)
        h_enc = self.encoder(x_t)  # (T_past,B,d_model)
        h = h_enc.mean(0)          # (B,d_model)

        # autoregressive decode
        mu_seq, logvar_seq = [], []
        inp = None
        for t in range(self.T_future):
            # decide input to GRU
            if future is not None:
                coord = future[:,t]             # (B,2)
                inp = self.input_proj(coord)    # teacher forcing
            else:
                if t == 0:
                    inp = self.input_proj(past[:, -1])
                else:
                    inp = self.input_proj(mu_seq[-1])            # previous mu
            # update hidden
            h = self.gru(inp, h)               # (B,d)
            # predict params
            params = self.out_head(h)          # (B,4)
            mu_t = params[:,:2]
            logvar_t = params[:,2:]
            mu_seq.append(mu_t)
            logvar_seq.append(logvar_t)
        mu      = torch.stack(mu_seq, dim=1)      # (B, T_future,2)
        logvar  = torch.stack(logvar_seq, dim=1)  # (B, T_future,2)
        return mu, logvar


def gaussian_nll_loss(mu, logvar, target, occ_mask):
    """
    Diagonal Gaussian NLL, masked by occ_mask (1=visible).
    mu, logvar: (B, T_future,2)
    target:     (B, T_future,2)
    occ_mask:   (B, T_future)
    """
    var = torch.exp(logvar)
    diff2 = (target - mu)**2
    nll = 0.5*(logvar + diff2/var)
    mask = occ_mask.unsqueeze(-1)
    nll = nll * mask
    total_vis = mask.sum(dim=[1,2]).clamp_min(1.0)
    loss = nll.sum(dim=[1,2]) / total_vis
    return loss.mean()

# End of code file


def train(
    drone_data_root: str,
    original_dataset_root: str,
    classes: list,
    T_past: int = 10,
    T_future: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 20,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    d_set = AgentSequenceDataset(drone_data_root=drone_data_root, original_dataset_root=original_dataset_root, classes=classes, T_past=T_past, T_future=T_future)
    train_dataloader = DataLoader(d_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    model = TrajectoryModel(10, 10, d_model = 64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min = 1e-5)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            past     = batch['past'].to(device)      # (B, T_past, 2)
            obs_mask = batch['obs_mask'].to(device)  # (B, T_past)
            future   = batch['future'].to(device)    # (B, T_future, 2)
            occ_mask = batch['occ_mask'].to(device)  # (B, T_future)

            mu, logvar = model(past, obs_mask, future=future)
            loss = gaussian_nll_loss(mu, logvar, future, occ_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * past.size(0)
        scheduler.step()

        running_loss = running_loss/len(d_set)
    torch.save(model.state_dict(), "trajectory_model_final.pth")
    print("Training complete. Model saved to trajectory_model_final.pth")


if __name__ == "__main__":
    train(
        drone_data_root="square_stanford_data",
        original_dataset_root="stanford_data/archive",
        classes=['Pedestrian','Biker','Skater','Cart','Car','Bus'],
        T_past=10,
        T_future=10,
        batch_size=64,
        lr=1e-3,
        epochs=20
    )