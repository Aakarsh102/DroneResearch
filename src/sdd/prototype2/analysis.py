# import os
# import pandas as pd

# def find_max_agents_in_frame(original_dataset_root, classes=None):
#     """
#     Scans all annotation files under
#         original_dataset_root/annotations/<loc>/<vid>/annotations.txt
#     and finds the frame with the maximum number of distinct agents (trackIds).

#     Args:
#         original_dataset_root (str): path to your dataset root
#         classes (list[str] or None): if provided, only count agents whose
#             'label' is in this list

#     Returns:
#         max_count (int): maximum number of agents in any single frame
#         max_info (tuple): (location, video, frame_number) where it occurred
#     """
#     annotations_root = os.path.join(original_dataset_root, "annotations")
#     max_count = 0
#     max_info = (None, None, None)

#     for loc in os.listdir(annotations_root):
#         loc_path = os.path.join(annotations_root, loc)
#         if not os.path.isdir(loc_path):
#             continue

#         for vid in os.listdir(loc_path):
#             vid_path = os.path.join(loc_path, vid)
#             if vid_path.endswith("deathCircle/video1"):continue
#             if vid_path.endswith("deathCircle/video3"):continue
#             ann_file = os.path.join(vid_path, "annotations.txt")
#             if not os.path.isfile(ann_file):
#                 continue

#             # Read only trackId, frame (and label if filtering)
#             usecols = ['trackId', 'frame']
#             if classes is not None:
#                 usecols.append('label')

#             df = pd.read_csv(
#                 ann_file, sep=' ', header=None,
#                 names=['trackId','xmin','ymin','xmax','ymax',
#                        'frame','lost','occluded','generated','label'],
#                 usecols=usecols
#             )

#             # Optionally filter by class
#             if classes is not None:
#                 df = df[df['label'].isin(classes)]

#             # Count unique agents per frame

#             counts = df.groupby('frame')['trackId'].nunique()

#             if counts.empty:
#                 continue

#             local_max = int(counts.max())
#             if local_max > max_count:
#                 max_count = local_max
#                 max_frame = int(counts.idxmax())
#                 max_info = (loc, vid, max_frame)

#     print(f"→ Maximum agents in any frame: {max_count}")
#     print(f"   (location='{max_info[0]}', video='{max_info[1]}', frame={max_info[2]})")
#     return max_count, max_info

# if __name__ == "__main__":
#     # Example usage:
#     root = "/Users/aakarshrai/Desktop/stanford_data/archive"
#     # If you only care about certain classes, pass a list, e.g. ['Car','Pedestrian']
#     find_max_agents_in_frame(root, classes=None)

import os
import pandas as pd

def find_max_agents_window(original_dataset_root, classes=None, window_length=240):
    """
    Scans all annotation files under
        original_dataset_root/annotations/<loc>/<vid>/annotations.txt
    and finds the contiguous `window_length`-frame interval in which the
    number of agents **present in every frame** of that interval is maximized.

    Args:
        original_dataset_root (str): path to your dataset root
        classes (list[str] or None): if provided, only count agents whose
            'label' is in this list
        window_length (int): number of consecutive frames in each window

    Returns:
        max_count (int): maximum number of agents present in *all* frames of any window
        max_info (tuple): (location, video, start_frame) where it occurred
    """
    annotations_root = os.path.join(original_dataset_root, "annotations")
    global_max = 0
    global_info = (None, None, None)

    for loc in os.listdir(annotations_root):
        loc_path = os.path.join(annotations_root, loc)
        if not os.path.isdir(loc_path):
            continue

        for vid in os.listdir(loc_path):
            vid_path = os.path.join(loc_path, vid)
            # Skip problematic videos if needed
            if vid_path.endswith("deathCircle/video1") or vid_path.endswith("deathCircle/video3"):
                continue

            ann_file = os.path.join(vid_path, "annotations.txt")
            if not os.path.isfile(ann_file):
                continue

            # Read only necessary columns
            usecols = ['trackId', 'frame']
            if classes is not None:
                usecols.append('label')

            try:
                df = pd.read_csv(
                    ann_file, sep=' ', header=None,
                    names=['trackId','xmin','ymin','xmax','ymax',
                           'frame','lost','occluded','generated','label'],
                    usecols=usecols
                )
            except Exception as e:
                print(f"Error reading {ann_file}: {e}")
                continue
                
            if df.empty:
                continue
                
            # Filter by classes if specified
            if classes is not None:
                df = df[df['label'].isin(classes)]
                if df.empty:
                    continue

            # Build frame -> set of trackIds mapping
            frame_to_ids = df.groupby('frame')['trackId'].apply(set).to_dict()
            
            # Get frame range
            frames = sorted(frame_to_ids.keys())
            if len(frames) < window_length:
                continue
            
            # Find all possible consecutive frame sequences of window_length
            max_local = 0
            best_start = None
            
            # Check all possible starting positions
            for i in range(len(frames) - window_length + 1):
                start_frame = frames[i]
                end_frame = frames[i + window_length - 1]
                
                # Check if we have a consecutive sequence
                window_frames = frames[i:i + window_length]
                expected_frames = list(range(start_frame, start_frame + window_length))
                
                # Only consider truly consecutive frame sequences
                if window_frames == expected_frames:
                    # Find intersection of all agents across all frames in this window
                    agents_in_window = None
                    
                    for frame in window_frames:
                        frame_agents = frame_to_ids.get(frame, set())
                        if agents_in_window is None:
                            agents_in_window = frame_agents.copy()
                        else:
                            agents_in_window = agents_in_window.intersection(frame_agents)
                    
                    intersection_count = len(agents_in_window) if agents_in_window else 0
                    
                    if intersection_count > max_local:
                        max_local = intersection_count
                        best_start = start_frame

            # Update global maximum
            if max_local > global_max:
                global_max = max_local
                global_info = (loc, vid, best_start)
                
            print(f"Processed {loc}/{vid}: max agents = {max_local}")

    print(f"\n→ Maximum agents persistently present over any {window_length}-frame window: {global_max}")
    if global_info[0] is not None:
        print(f"   (location='{global_info[0]}', video='{global_info[1]}', start_frame={global_info[2]})")
    else:
        print("   No valid windows found.")
    
    return global_max, global_info


def find_max_agents_window_optimized(original_dataset_root, classes=None, window_length=240):
    """
    Optimized version using sliding window technique for better performance.
    This version is more efficient for large datasets.
    """
    annotations_root = os.path.join(original_dataset_root, "annotations")
    global_max = 0
    global_info = (None, None, None)

    for loc in os.listdir(annotations_root):
        loc_path = os.path.join(annotations_root, loc)
        if not os.path.isdir(loc_path):
            continue

        for vid in os.listdir(loc_path):
            vid_path = os.path.join(loc_path, vid)
            if vid_path.endswith("deathCircle/video1") or vid_path.endswith("deathCircle/video3"):
                continue

            ann_file = os.path.join(vid_path, "annotations.txt")
            if not os.path.isfile(ann_file):
                continue

            usecols = ['trackId', 'frame']
            if classes is not None:
                usecols.append('label')

            try:
                df = pd.read_csv(
                    ann_file, sep=' ', header=None,
                    names=['trackId','xmin','ymin','xmax','ymax',
                           'frame','lost','occluded','generated','label'],
                    usecols=usecols
                )
            except Exception as e:
                print(f"Error reading {ann_file}: {e}")
                continue
                
            if df.empty:
                continue
                
            if classes is not None:
                df = df[df['label'].isin(classes)]
                if df.empty:
                    continue

            # Create a complete frame range and agent presence matrix
            min_frame = int(df['frame'].min())
            max_frame = int(df['frame'].max())
            
            if max_frame - min_frame + 1 < window_length:
                continue
            
            # Build presence matrix: frame -> set of trackIds
            frame_agents = {}
            for frame in range(min_frame, max_frame + 1):
                frame_agents[frame] = set()
            
            for _, row in df.iterrows():
                frame_agents[int(row['frame'])].add(row['trackId'])
            
            # Find maximum intersection using sliding window
            max_local = 0
            best_start = None
            
            # For each possible window start position
            for start in range(min_frame, max_frame - window_length + 2):
                # Find agents present in ALL frames of this window
                window_intersection = None
                
                for frame in range(start, start + window_length):
                    if window_intersection is None:
                        window_intersection = frame_agents[frame].copy()
                    else:
                        window_intersection = window_intersection.intersection(frame_agents[frame])
                        # Early termination if intersection becomes empty
                        if not window_intersection:
                            break
                
                intersection_count = len(window_intersection) if window_intersection else 0
                
                if intersection_count > max_local:
                    max_local = intersection_count
                    best_start = start

            # Update global maximum
            if max_local > global_max:
                global_max = max_local
                global_info = (loc, vid, best_start)
                
            print(f"Processed {loc}/{vid}: max agents = {max_local}")

    print(f"\n→ Maximum agents persistently present over any {window_length}-frame window: {global_max}")
    if global_info[0] is not None:
        print(f"   (location='{global_info[0]}', video='{global_info[1]}', start_frame={global_info[2]})")
    else:
        print("   No valid windows found.")
    
    return global_max, global_info


if __name__ == "__main__":
    root = "/Users/aakarshrai/Desktop/stanford_data/archive"
    
    # Use the optimized version for better performance
    max_count, info = find_max_agents_window_optimized(root, classes=None, window_length=1)
    
    # For testing with window_length=1 (equivalent to finding max agents in any single frame)
    # max_count, info = find_max_agents_window_optimized(root, classes=None, window_length=1)
