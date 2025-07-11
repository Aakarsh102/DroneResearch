import cv2 
import os
new_size = (2048, 2048)
root_dir = "stanford_data/archive/annotations"
drone_dir = "square_stanford_data/"

for location in os.listdir(root_dir):
    location_dir = os.path.join(root_dir, location)
    drone_location = os.path.join(drone_dir, location)
    if not os.path.isdir(location_dir):
        continue

    for video in os.listdir(location_dir):
        video_dir = os.path.join(location_dir, video)
        if not os.path.isdir(video_dir):
            continue

        ref_path = os.path.join(video_dir, "reference.jpg")
        img = cv2.imread(ref_path)
        if img is None:
            print("img not found")
            continue
        print(img.shape)

        rx = new_size[1] / img.shape[1]
        ry = new_size[0] / img.shape[0]
        img = cv2.resize(img, new_size, cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(video_dir, "reference1.jpg"))



