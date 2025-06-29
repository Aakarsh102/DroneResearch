import pandas as pd
import matplotlib.pyplot as plt

# Path to your Argoverse CSV file
path = "data/val/data/1.csv"  # Change to any file in val/train/test

# Load the CSV
df = pd.read_csv(path)

# Get unique track ids and filter for agent
agents = df[df['OBJECT_TYPE'] == 'AGENT']
others = df[df['OBJECT_TYPE'] == 'OTHERS']

# Plotting
plt.figure(figsize=(10, 10))

# Plot other objects in gray
for tid in others['TRACK_ID'].unique():
    coords = others[others['TRACK_ID'] == tid][['X', 'Y']]
    plt.plot(coords['X'], coords['Y'], color='lightgray', linewidth=1)

# Plot the agent in red
agent_coords = agents[['X', 'Y']]
plt.plot(agent_coords['X'], agent_coords['Y'], color='red', linewidth=2, label='Agent')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Argoverse Forecasting Trajectories")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
