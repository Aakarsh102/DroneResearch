import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random, cv2, matplotlib.pyplot as plt
import torch
from new_model_training import TrajectoryModel, gaussian_nll_loss              # your network     
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
    'label': idx,
    'loc': loc,         #  add this
    'vid': vid          #  add this
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
            'label':    torch.tensor(s['label'],dtype=torch.long),
            'loc': s['loc'],
            'vid': s['vid']
        }


drone_root   = "/Users/aakarshrai/Desktop/square_stanford_data"
orig_root    = "/Users/aakarshrai/Desktop/stanford_data/archive"          # original dataset path (w/ reference.jpg)
classes      = ['Pedestrian','Biker','Skater','Cart','Car','Bus']
T_past, T_fut= 10, 10
ckpt_path    = "/Users/aakarshrai/Desktop/trajectory_model_final1.pth"
device       = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Dataset & model ------------------------------------------------------
ds = AgentSequenceDataset(drone_root, orig_root, classes, T_past, T_fut)
model = TrajectoryModel(T_past, T_fut, d_model=64).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# 3. Pick one sample ------------------------------------------------------
idx = random.randrange(len(ds))
sample = ds[idx]
past      = sample['past'].unsqueeze(0).to(device)      # (1,T_past,2)
obs_mask  = sample['obs_mask'].unsqueeze(0).to(device)  # (1,T_past)
gt_future = sample['future'].numpy()                    # (T_future,2)
loc, vid  = sample['loc'], sample['vid']                # need these from dataset

# 4. Inference (no teacher forcing) --------------------------------------
with torch.no_grad():
    mu_pred, _ = model(past, obs_mask, future=None)     # (1,T_future,2)
pred_future = mu_pred.squeeze(0).cpu().numpy()          # (T_future,2)
past_np     = past.squeeze(0).cpu().numpy()             # (T_past,2)

# 5. Load reference image -------------------------------------------------
ref_path = f"{orig_root}/annotations/{loc}/{vid}/reference.jpg"
img = cv2.imread(ref_path)
if img is None:
    raise FileNotFoundError(f"Cannot load {ref_path}")
canvas = img.copy()

# 6. Draw trajectories ----------------------------------------------------
def draw_polyline(img, pts, color, thickness=4):
    pts = pts.astype(int)
    for i in range(len(pts)-1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), color, thickness)

draw_polyline(canvas, past_np,        (0,0,255))   # blue  (BGR)
draw_polyline(canvas, pred_future,    (0,0,255),1) # red
draw_polyline(canvas, gt_future,      (0,255,0),1) # green

cv2.circle(canvas, tuple(past_np[-1].astype(int)), 4, (255,255,0), -1)   # last past
cv2.circle(canvas, tuple(gt_future[-1].astype(int)), 4, (0,255,0), -1)   # GT end
cv2.circle(canvas, tuple(pred_future[-1].astype(int)),4,(0,0,255), -1)   # pred end

# 7. Save & display -------------------------------------------------------
out_file = "demo_overlay.jpg"
cv2.imwrite(out_file, canvas)
print(f"Overlay saved to {out_file}")

plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f"{loc}/{vid}  —  sample #{idx}")
plt.show()