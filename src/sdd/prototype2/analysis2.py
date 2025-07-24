


# #!/usr/bin/env python3
# import cv2
# import pandas as pd
# import argparse
# import os

# def overlay_bboxes_on_video(video_path, ann_path, output_path, classes=None, show=False):
#     # 1) Load annotations
#     cols = ['trackId','xmin','ymin','xmax','ymax','frame','lost','occluded','generated','label']
#     df = pd.read_csv(
#         ann_path, sep=' ', header=None, names=cols,
#         usecols=['trackId','xmin','ymin','xmax','ymax','frame','label']
#     )
#     if classes:
#         df = df[df['label'].isin(classes)]
#     boxes_by_frame = {
#         frame: grp[['trackId','xmin','ymin','xmax','ymax','label']].to_dict('records')
#         for frame, grp in df.groupby('frame')
#     }

#     # 2) Open video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {video_path}")
#     fps    = cap.get(cv2.CAP_PROP_FPS)
#     w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # 3) Prepare output path
#         # 3) Prepare output path: allow passing a directory (existing or new) or a filepath
#     base_name, ext = os.path.splitext(output_path)
#     if os.path.isdir(output_path) or ext == "":
#         # treat as directory
#         out_dir = output_path if os.path.isdir(output_path) else output_path
#         os.makedirs(out_dir, exist_ok=True)
#         video_base = os.path.splitext(os.path.basename(video_path))[0]
#         output_file = os.path.join(out_dir, f"{video_base}_bboxes.mp4")
#     else:
#         # treat as file path
#         os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
#         output_file = output_path

#     # 4) Create VideoWriter with MP4v
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
#     if not writer.isOpened():
#         raise RuntimeError(f"Cannot write output: {output_file}")

#     # 5) Process each frame
#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         for box in boxes_by_frame.get(frame_idx, []):
#             x1, y1 = int(box['xmin']), int(box['ymin'])
#             x2, y2 = int(box['xmax']), int(box['ymax'])
#             tid    = int(box['trackId'])
#             lbl    = box['label']
#             cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
#             cv2.putText(
#                 frame, f"{lbl}:{tid}",
#                 (x1, max(0,y1-5)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA
#             )

#         writer.write(frame)
#         if show:
#             cv2.imshow('bboxes', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         frame_idx += 1

#     cap.release()
#     writer.release()
#     if show:
#         cv2.destroyAllWindows()

#     print(f"✅ Rendered {frame_idx} frames to {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Overlay bounding boxes onto a video")
#     parser.add_argument("--video",  required=True, help="Path to input video file")
#     parser.add_argument("--ann",    required=True, help="Path to annotations.txt")
#     parser.add_argument("--output", required=True,
#                         help="Output file or directory")
#     parser.add_argument("--classes", nargs="+", default=None,
#                         help="Optional list of labels to include")
#     parser.add_argument("--show", action="store_true",
#                         help="Display frames live (press 'q' to quit)")
#     args = parser.parse_args()

#     overlay_bboxes_on_video(
#         video_path=args.video,
#         ann_path=args.ann,
#         output_path=args.output,
#         classes=args.classes,
#         show=args.show
#     )


import cv2

# Path to your video
video_path = '/Users/aakarshrai/Desktop/video/deathCircle/video1/video.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

# Original and target frame‐rates
orig_fps = cap.get(cv2.CAP_PROP_FPS)       # should be 30
target_fps = 4

# Compute how many original frames to skip between displays
skip = int(round(orig_fps / target_fps))   # 30 / 2.5 ≈ 12

frame_idx = 0
delay_ms = int(1000 / target_fps)          # ~400 ms between frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only show every 12th frame
    if frame_idx % skip == 0:
        cv2.imshow('Video @2.5fps', frame)
        # quit if you press 'q'
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
