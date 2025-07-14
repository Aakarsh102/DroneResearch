import torch.nn as nn
import numpy as np
import os


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

        # Precompute obs sets per (loc,vid)
        obs: dict[tuple[str,str], set[tuple[int,int]]] = {}
        for loc in os.listdir(drone_data_root):
            loc_path = os.path.join(drone_data_root, loc)
            if not os.path.isdir(loc_path):
                continue
            for dr in os.listdir(loc_path):
                dr_path = os.path.join(loc_path, dr)
                if not os.path.isdir(dr_path):
                    continue
                for fname in os.listdir(dr_path):
                    if not fname.endswith("_annotations.txt"):
                        continue
                    vid = fname.replace("_annotations.txt","")
                    key = (loc, vid)
                    df = pd.read_csv(
                        os.path.join(dr_path,fname),
                        sep=' ', header=None,
                        names=['trackId',…,'frame',…,'label']
                    )
                    # vectorized set creation:
                    pairs = zip(df['trackId'].values,
                                df['frame'].astype(int).values)
                    obs[key] = set(pairs)

        # …then later you do `seen = obs.get(key, set())`
        # and your obs_mask = [1 if (tid,f) in seen else 0 for f in frames]
        # remains exactly the same.
