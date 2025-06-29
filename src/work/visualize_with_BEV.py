import pandas as pd
import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap

csv_path = "data/val/data/1.csv"
df = pd.read_csv(csv_path)
city_name = df["CITY_NAME"].iloc[0]

# Initialize ArgoverseMap
am = ArgoverseMap(root="argoverse-api/map_files")

# Plot map lane centerlines
lane_segments = am.city_lane_centerlines_dict[city_name]
# print(lane_segments)
# print("************************")
# for i in lane_segments:
#     print(i, lane_segments[i].centerline)
#     print("###########################")

for lane_id, lane_segment in lane_segments.items():
    centerline = lane_segment.centerline
    xs = [pt[0] for pt in centerline]
    ys = [pt[1] for pt in centerline]
    plt.plot(xs, ys, color='lightgray', linewidth=1)
    break

# Plot agent and other vehicles
for tid in df["TRACK_ID"].unique():
    obj = df[df["TRACK_ID"] == tid]
    if obj["OBJECT_TYPE"].iloc[0] == "AGENT":
        plt.plot(obj["X"], obj["Y"], "r-", label="AGENT", linewidth=2)
    else:
        plt.plot(obj["X"], obj["Y"], "gray", alpha=0.3)

plt.title("Forecasting Trajectories with Map")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.legend()
plt.show()
 
