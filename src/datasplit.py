import numpy as np
import pandas as pd 

def create_camera_data():
    df = pd.read_csv("orbit_simulation_1000_objects_positions.csv")
    # print(df.iloc[0:5])  # Print the first 5 rows of the DataFrame
    df['x'] = df['x'].apply(lambda val: int(val * 1000) / 1000)
    df['y'] = df['y'].apply(lambda val: int(val * 1000) / 1000)
    df['radius'] = df['radius'].apply(lambda val: int(val * 1000) / 1000)
    df['angular_velocity'] = df['angular_velocity'].apply(lambda val: int(val * 1000) / 1000)
    df1 = df[(df["x"] >= 0) & (df["y"] >= 0)]
    df2 = df[(df["x"] < 0)  & (df["y"] >= 0)]
    df3 = df[(df["x"] < 0)  & (df["y"] < 0)]
    df4 = df[(df["x"] >= 0) & (df["y"] < 0)]
    df1 = df1[['time_step', 'object_id', 'x', 'y']]
    df2 = df2[['time_step', 'object_id', 'x', 'y']]
    df3 = df3[['time_step', 'object_id', 'x', 'y']]
    df4 = df4[['time_step', 'object_id', 'x', 'y']]
    df1.to_csv("camera_data_1.csv", index=False)
    df2.to_csv("camera_data_2.csv", index=False)
    df3.to_csv("camera_data_3.csv", index=False)
    df4.to_csv("camera_data_4.csv", index=False)

if __name__ == "__main__":
    create_camera_data()
    print("Camera data created successfully.")