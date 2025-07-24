import os

def convert_annotations_fps(input_file, output_file, orig_fps=30, target_fps=2.5):
    """
    Convert annotations.txt file from original fps to target fps.
    Keeps every skip-th frame and renumbers frames continuously.
    
    Args:
        input_file: Path to original annotations.txt
        output_file: Path to save converted annotations.txt
        orig_fps: Original frame rate (default: 30)
        target_fps: Target frame rate (default: 2.5)
    """
    
    # Calculate skip factor (same as video conversion)
    skip = int(round(orig_fps / target_fps))  # 30 / 2.5 ≈ 12
    print(f"Converting from {orig_fps}fps to {target_fps}fps")
    print(f"Keeping every {skip}th frame")
    
    # Read all annotations
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse annotations and group by frame
    annotations_by_frame = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:  # Ensure we have at least frame info
            frame_num = int(parts[5])  # Frame is the 6th column (index 5)
            if frame_num not in annotations_by_frame:
                annotations_by_frame[frame_num] = []
            annotations_by_frame[frame_num].append(line.strip())
    
    # Get sorted frame numbers
    all_frames = sorted(annotations_by_frame.keys())
    print(f"Original frame range: {min(all_frames)} to {max(all_frames)}")
    print(f"Total original frames with annotations: {len(all_frames)}")
    
    # Keep every skip-th frame and renumber
    kept_annotations = []
    new_frame_num = 0
    
    for i, original_frame in enumerate(all_frames):
        # Keep every skip-th frame (same logic as video)
        if i % skip == 0:
            # Process all annotations for this frame
            for annotation in annotations_by_frame[original_frame]:
                parts = annotation.split()
                # Replace the frame number (index 5) with new continuous frame number
                parts[5] = str(new_frame_num)
                # Reconstruct the line
                new_annotation = ' '.join(parts)
                kept_annotations.append(new_annotation)
            
            new_frame_num += 1
    
    print(f"Kept {new_frame_num} frames (every {skip}th frame)")
    print(f"New frame range: 0 to {new_frame_num - 1}")
    print(f"Total kept annotations: {len(kept_annotations)}")
    
    # Write converted annotations
    with open(output_file, 'w') as f:
        for annotation in kept_annotations:
            f.write(annotation + '\n')
    
    print(f"Converted annotations saved to: {output_file}")

def process_all_annotations(base_dir):
    """
    Process all annotations.txt files in the directory structure.
    
    Args:
        base_dir: Base directory containing the folder structure
    """
    
    processed_count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        if 'annotations.txt' in files:
            input_file = os.path.join(root, 'annotations.txt')
            output_file = os.path.join(root, 'annotations_2.5fps.txt')
            
            print(f"\nProcessing: {input_file}")
            try:
                convert_annotations_fps(input_file, output_file)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
    
    print(f"\nProcessed {processed_count} annotation files.")

# Example usage
if __name__ == "__main__":
    # Option 1: Process a single annotations.txt file
    # convert_annotations_fps('path/to/annotations.txt', 'path/to/annotations_2.5fps.txt')
    
    # Option 2: Process all annotations.txt files in directory structure
    base_directory = "."  # Change this to your base directory path
    process_all_annotations(base_directory)
    
    # Option 3: Process specific file (uncomment and modify path)
    # convert_annotations_fps('./deathCircle/video1/annotations.txt', 
    #                        './deathCircle/video1/annotations_2.5fps.txt')