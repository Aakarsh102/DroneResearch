import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap

import os
import glob
import pandas as pd
import numpy as np

from math import cos, sin, hypot

class Drone:
    """
    Simple drone model:
      - id:            unique identifier
      - path:          pd.DataFrame with columns ['TIMESTAMP','X','Y'] (and optional 'YAW' in radians)
      - fov_radius:    scalar in same units as X,Y (meters)
    """
    def __init__(self, drone_id: str, path: pd.DataFrame, fov_radius: float):
        self.id = drone_id
        self.path = path.copy()
        if 'YAW' not in self.path.columns:
            # assume north-facing (yaw=0) if not provided
            self.path['YAW'] = 0.0
        self.fov_radius = fov_radius


def transform_to_local(gx, gy, dx, dy, yaw):
    """
    Rotate & translate global (gx,gy) into a drone‐centric frame:
       origin at (dx,dy), +x forward along drone yaw=0, +y to left.
    """
    dxg = gx - dx
    dyg = gy - dy
    # rotate by -yaw
    c, s = cos(-yaw), sin(-yaw)
    lx = dxg * c - dyg * s
    ly = dxg * s + dyg * c
    return lx, ly


def simulate_drones(
    objects_df: pd.DataFrame,
    drones: list[Drone],
    output_dir: str = "drone_outputs",
    one_file_per_drone: bool = True
):
    """
    For each Drone, for each timestamp in its path, find all objects
    in objects_df at that timestamp, transform their coordinates into
    the drone frame, filter by fov_radius, and save to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_records = []

    for drone in drones:
        # join objects with this drone’s path on TIMESTAMP
        df = objects_df.merge(
            drone.path[['TIMESTAMP','X','Y','YAW']],
            on='TIMESTAMP',
            suffixes=('','_drone')
        )
        # print(df.columns())

        # compute local coords
        local_coords = df.apply(
            lambda r: transform_to_local(
                r['X'], r['Y'],
                r['X_drone'], r['Y_drone'],
                r['YAW']
            ),
            axis=1,
            result_type='expand'
        )
        df['local_x'], df['local_y'] = local_coords[0], local_coords[1]

        # filter by circular FOV
        df = df[df.apply(lambda r: hypot(r['local_x'], r['local_y']) <= drone.fov_radius, axis=1)]

        # record for output
        df_out = df[[
            'TIMESTAMP','TRACK_ID','OBJECT_TYPE',
            'X','Y','local_x','local_y'
        ]].copy()
        df_out['DRONE_ID'] = drone.id

        if one_file_per_drone:
            fname = os.path.join(output_dir, f"drone_{drone.id}_tracks.csv")
            df_out.to_csv(fname, index=False)
            print(f"Saved {len(df_out)} detections → {fname}")
        else:
            all_records.append(df_out)

    if not one_file_per_drone:
        merged = pd.concat(all_records, ignore_index=True)
        fname = os.path.join(output_dir, "all_drones_tracks.csv")
        merged.to_csv(fname, index=False)
        print(f"Saved {len(merged)} total detections → {fname}")


if __name__ == "__main__":
    # 1. Load all Argoverse CSVs (e.g. train + val)
    files = glob.glob("data/train/data/*.csv", recursive=True)
    df_list = [pd.read_csv(f) for f in files]
    objects = pd.concat(df_list, ignore_index=True)

    # 2. Define two example drones with simple straight‐line paths
    #    (in practice replace these with your real flight logs)
    timestamps = sorted(objects['TIMESTAMP'].unique())
    # Drone A flies east along y=2300 from t0→tN
    pathA = pd.DataFrame({
        'TIMESTAMP': timestamps,
        'X': np.linspace(4100, 4500, len(timestamps)),
        'Y': 2300,
    })
    # Drone B hovers at one point
    pathB = pd.DataFrame({
        'TIMESTAMP': timestamps,
        'X': 4300,
        'Y': 2400,
    })

    droneA = Drone("A", pathA, fov_radius=100.0)
    droneB = Drone("B", pathB, fov_radius=120.0)

    # 3. Run the sim and save CSVs
    simulate_drones(objects, [droneA, droneB], output_dir="drone_outputs", one_file_per_drone=True)
