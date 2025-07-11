import os
import numpy as np
import pandas as pd
import json
import cv2
import torch
from torch.utils.data import Dataset

class AgentSequenceDataset(Dataset):
    """
    PyTorch Dataset yielding per-agent trajectory sequences.

    For each agent track, returns:
      - past:  Tensor[T_past, 2]  (x,y) positions
      - future:Tensor[T_future, 2] (x,y) positions
      - label: Tensor               class index
    Only sequences where both past and future windows are fully observed are included.
    """
    def __init__(self,
                 drone_data_root: str,
                 original_dataset_root: str,
                 classes: list,
                 T_past:int = 10,
                 T_future:int = 10):
        self.drone_data_root = drone_data_root
        self.orig_root = original_dataset_root
        self.classes = classes
        self.cls2idx = {c: i for i, c in enumerate(classes)}
        self.T_past = T_past
        self.T_future = T_future
        self.samples = [] 

        for loc in os.listdir(self.drone_data_root):
            loc_path = os.path.join(self.drone_data_root, loc)
            drone1_dir = os.path.join(loc_path, "drone1")
            vids = [filen.replace("_annotations.txt", "") for filen in os.listdir(drone1_dir) if filen.endswith("_annotations.txt")]
            for vid in vids:
                dfs = []
                for dir in os.listdir(loc_path):
                    ann_file = os.path.join(loc_path, dir, f"{vid}_annotations.txt")
                    if not os.path.isfile(ann_file):
                        continue
                    df = pd.read_csv(ann_file, sep = ' ', header = None, 
                                     names=['trackId','xmin','ymin','xmax','ymax','frame','lost','occluded','generated','label'])
                    df['x'] = (df.xmin + df.xmax) / 2
                    df['y'] = (df.xmin + df.xmax) / 2
                    dfs.append(df[['trackId','frame','x','y','label']])
                if not dfs:
                    continue
                data = pd.concat(dfs, ignore_index=True)
                data = data.drop_duplicates(subset=['trackId', 'frame'])
                for trackId, grp in data.groupby('trackId'):
                    grp = grp.sort_values('frame')
                    frames = grp['frame'].values
                    coords = np.vstack([grp['x'].values, grp['y'].values]).T
                    labels = grp['label'].values

                    # sliding window
                    L = len(frames)
                    window = self.T_past + self.T_future
                    for i in range(L - window + 1):
                        # check continuous frames
                        if frames[i+self.T_past-1] - frames[i] != self.T_past-1:
                            continue
                        if frames[i+window-1] - frames[i+self.T_past] != self.T_future-1:
                            continue
                        past = coords[i:i+self.T_past]
                        future = coords[i+self.T_past:i+window]
                        # label at last past step
                        lbl = labels[i+self.T_past-1]
                        if lbl not in self.cls2idx:
                            continue
                        self.samples.append({
                            'past': past.astype(np.float32),
                            'future': future.astype(np.float32),
                            'label': self.cls2idx[lbl]
                        })
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'past':   torch.from_numpy(s['past']),    # (T_past, 2)
            'future': torch.from_numpy(s['future']),  # (T_future, 2)
            'label':  torch.tensor(s['label'], dtype=torch.long)
        }

                    