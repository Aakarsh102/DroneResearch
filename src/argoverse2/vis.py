# enhanced_av2_vis.py

import numpy as np

# ── patch for AV2's typing (removes np.bool errors) ──
if not hasattr(np, "bool"):
    np.bool = bool

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from pathlib import Path
import seaborn as sns

from av2.datasets.motion_forecasting.scenario_serialization import \
    load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap

# ── CONFIG ────────────────────────────────────────────────────────────────
SCENARIO_ROOT = Path("../../argoverse2_data/val")  # your test/ folder
OUT_DIR = Path("claude_videos")
OUT_DIR.mkdir(exist_ok=True)

# Use matplotlib colormap for agent diversity
cmap = plt.colormaps["tab20"]

# Color scheme for different agent types
AGENT_COLORS = {
    'focal_track': '#FF0000',  # Red for focal agent
    'vehicle': '#1f77b4',  # Blue for vehicles
    'pedestrian': '#ff7f0e',  # Orange for pedestrians
    'motorcyclist': '#2ca02c',  # Green for motorcyclists
    'cyclist': '#d62728',  # Dark red for cyclists
    'bus': '#9467bd',  # Purple for buses
    'static': '#8c564b',  # Brown for static objects
    'background_vehicle': '#17becf',  # Cyan for background vehicles
    'unknown': '#7f7f7f'  # Gray for unknown
}

# Map AV2 object categories to our color scheme
CATEGORY_MAP = {
    'vehicle': 'vehicle',
    'pedestrian': 'pedestrian',
    'motorcyclist': 'motorcyclist',
    'cyclist': 'cyclist',
    'bus': 'bus',
    'static': 'static',
    'background_vehicle': 'background_vehicle',
    'unknown': 'unknown'
}


# ────────────────────────────────────────────────────────────────────────────

def get_agent_color(track_idx, focal_track_id, track_id, cmap):
    """Get color for an agent based on whether it's focal or use colormap"""
    if track_id == focal_track_id:
        return AGENT_COLORS['focal_track']
    else:
        return cmap(track_idx % cmap.N)


def get_agent_size(track_id, focal_track_id):
    """Get marker size based on whether it's focal agent"""
    if track_id == focal_track_id:
        return 120  # Larger for focal agent
    else:
        return 60  # Standard size for other agents


def get_agent_marker(track_id, focal_track_id):
    """Get marker style based on whether it's focal agent"""
    if track_id == focal_track_id:
        return 's'  # Square for focal agent
    else:
        return 'o'  # Circle for other agents


def draw_agent_trajectory(ax, track_pos_map, color, alpha=0.3, linewidth=1):
    """Draw trajectory trail for an agent"""
    positions = list(track_pos_map.values())
    if len(positions) > 1:
        xs, ys = zip(*positions)
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, linestyle='--')


for scenario_dir in sorted(SCENARIO_ROOT.iterdir()):
    if not scenario_dir.is_dir():
        continue

    sid = scenario_dir.name
    pq_path = scenario_dir / f"scenario_{sid}.parquet"
    map_path = scenario_dir / f"log_map_archive_{sid}.json"
    if not pq_path.exists() or not map_path.exists():
        print(f"[!] Missing files for {sid}, skipping.")
        continue

    # ---- load the motion scenario and its matching map ----
    scenario = load_argoverse_scenario_parquet(pq_path)
    static_map = ArgoverseStaticMap.from_json(map_path)

    # ---- identify focal track ----
    focal_track_id = scenario.focal_track_id

    # ---- build per-track {timestep: (x,y)} dicts with metadata ----
    track_pos_maps = [
        {st.timestep: st.position for st in tr.object_states}
        for tr in scenario.tracks
    ]

    # Build track metadata
    track_data = []
    for i, (tr, pos_map) in enumerate(zip(scenario.tracks, track_pos_maps)):
        track_data.append({
            'track_id': tr.track_id,
            'pos_map': pos_map,
            'color': get_agent_color(i, focal_track_id, tr.track_id, cmap),
            'size': get_agent_size(tr.track_id, focal_track_id),
            'marker': get_agent_marker(tr.track_id, focal_track_id),
            'is_focal': tr.track_id == focal_track_id
        })

    num_frames = len(scenario.timestamps_ns)
    num_tracks = len(track_data)

    # ---- compute global xy bounds (with padding) ----
    all_pts = [pt for td in track_data for pt in td['pos_map'].values()]
    if not all_pts:
        print(f"[!] No position data for {sid}, skipping.")
        continue

    xs, ys = zip(*all_pts)
    pad = 15
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    # ---- set up plot with better styling ----
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", "box")
    ax.set_facecolor('white')  # Keep white background for better road visibility
    ax.grid(True, alpha=0.3)

    # ---- draw HD map with enhanced styling ----
    # Draw lane boundaries with better visibility
    for lane_seg in static_map.vector_lane_segments.values():
        left = lane_seg.left_lane_boundary.xyz[:, :2]
        right = lane_seg.right_lane_boundary.xyz[:, :2]

        # Draw lane boundaries with thicker lines
        ax.plot(left[:, 0], left[:, 1], color="black", lw=1.5, alpha=0.8)
        ax.plot(right[:, 0], right[:, 1], color="black", lw=1.5, alpha=0.8)

        # Fill lane area with light gray
        if len(left) > 1 and len(right) > 1:
            # Create a closed polygon for the lane
            try:
                lane_points = np.vstack([left, right[::-1]])
                lane_polygon = patches.Polygon(lane_points,
                                               facecolor='lightgray',
                                               alpha=0.3,
                                               edgecolor='none')
                ax.add_patch(lane_polygon)
            except:
                pass  # Skip if polygon creation fails

    # ---- draw crosswalks and other map elements ----
    for crosswalk in static_map.vector_pedestrian_crossings.values():
        polygon_pts = crosswalk.polygon[:, :2]
        crosswalk_patch = patches.Polygon(polygon_pts,
                                          facecolor='yellow',
                                          alpha=0.5,
                                          edgecolor='orange',
                                          linewidth=2)
        ax.add_patch(crosswalk_patch)

    # ---- prepare scatter and trail visualization ----
    scat = ax.scatter([], [], s=25)

    # Pre-create trail lines for each track
    trail_length = 15  # number of past frames to show
    trails = []
    for i in range(num_tracks):
        line, = ax.plot([], [], lw=2, alpha=0.6)
        trails.append(line)

    # Create a simple legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=AGENT_COLORS['focal_track'], markersize=10,
                   label='Focal Agent', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='blue', markersize=8,
                   label='Other Agents', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)


    def update(frame_idx):
        # Collect current positions, colors, and sizes
        pts = []
        colors = []
        sizes = []
        markers = []

        # Prepare trail data
        trail_segments = [np.empty((0, 2)) for _ in range(num_tracks)]

        for i, td in enumerate(track_data):
            if frame_idx in td['pos_map']:
                # Current position
                pts.append(td['pos_map'][frame_idx])
                colors.append(td['color'])
                sizes.append(td['size'])
                markers.append(td['marker'])

                # Collect trail positions
                past_positions = []
                for dt in range(trail_length):
                    t = frame_idx - dt
                    if t in td['pos_map']:
                        past_positions.append(td['pos_map'][t])

                if past_positions:
                    trail_segments[i] = np.array(past_positions)

        # Update scatter plot
        if pts:
            pts_arr = np.array(pts)
            scat.set_offsets(pts_arr)
            scat.set_color(colors)
            scat.set_sizes(sizes)
        else:
            scat.set_offsets([])

        # Update trails
        for i, (line, td) in enumerate(zip(trails, track_data)):
            seg = trail_segments[i]
            if seg.shape[0] > 0:
                line.set_data(seg[:, 0], seg[:, 1])
                line.set_color(td['color'])
                # Make focal agent trail more prominent
                if td['is_focal']:
                    line.set_linewidth(3)
                    line.set_alpha(0.8)
                else:
                    line.set_linewidth(1.5)
                    line.set_alpha(0.5)
            else:
                line.set_data([], [])

        # Update title
        ax.set_title(f"Scenario {sid} | Frame {frame_idx + 1}/{num_frames} | "
                     f"Time: {frame_idx * 0.1:.1f}s",
                     fontsize=14, pad=20)

        return (scat, *trails)


    # ---- animate & save MP4 ----
    print(f"[→] Creating animation for scenario {sid}...")
    ani = FuncAnimation(fig, update,
                        frames=num_frames,
                        interval=50,  # 20 Hz for smooth animation
                        blit=True)  # Enable blitting for performance

    out_file = OUT_DIR / f"enhanced_val_{sid}.mp4"
    ani.save(out_file, fps=20, dpi=150)
    plt.close(fig)
    print(f"[✓] Wrote {out_file}")

print(f"[✓] All videos saved to {OUT_DIR}")