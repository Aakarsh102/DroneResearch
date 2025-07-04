import numpy as np
# ── patch for AV2’s typing (removes np.bool errors) ──
if not hasattr(np, "bool"):
    np.bool = bool

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)
from av2.map.map_api import ArgoverseStaticMap

# ── CONFIG ────────────────────────────────────────────────────────────────
SCENARIO_ROOT = Path("../../argoverse2_data/val")  # your test/ folder
OUT_DIR       = Path("output_videos")
OUT_DIR.mkdir(exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────

# use the new API
cmap = plt.colormaps["tab20"]

for scenario_dir in sorted(SCENARIO_ROOT.iterdir()):
    if not scenario_dir.is_dir():
        continue

    sid      = scenario_dir.name
    pq_path  = scenario_dir / f"scenario_{sid}.parquet"
    map_path = scenario_dir / f"log_map_archive_{sid}.json"
    if not pq_path.exists() or not map_path.exists():
        print(f"[!] Missing files for {sid}, skipping.")
        continue

    # ---- load scenario + map ----
    scenario   = load_argoverse_scenario_parquet(pq_path)
    static_map = ArgoverseStaticMap.from_json(map_path)

    # ---- build per-track position dicts + assign colors ----
    track_pos_maps = [
        {st.timestep: st.position for st in tr.object_states}
        for tr in scenario.tracks
    ]
    num_tracks    = len(track_pos_maps)
    colors_by_idx = [cmap(i % cmap.N) for i in range(num_tracks)]
    num_frames    = len(scenario.timestamps_ns)

    # ---- compute global xy bounds (with padding) ----
    all_pts = [pt for pm in track_pos_maps for pt in pm.values()]
    xs, ys  = zip(*all_pts)
    pad = 10
    xmin, xmax = min(xs)-pad, max(xs)+pad
    ymin, ymax = min(ys)-pad, max(ys)+pad

    # ---- set up figure ----
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", "box")
    ax.set_title(f"Scenario {sid} — colored agents")

    # ---- draw HD map lanes ----
    for lane_seg in static_map.vector_lane_segments.values():
        left  = lane_seg.left_lane_boundary.xyz[:, :2]
        right = lane_seg.right_lane_boundary.xyz[:, :2]
        ax.plot(left[:,0],  left[:,1],  color="gray", lw=0.4, alpha=0.6)
        ax.plot(right[:,0], right[:,1], color="gray", lw=0.4, alpha=0.6)

    # ---- scatter for all agents ----
    scat = ax.scatter([], [], s=25)

    # ---- pre-create trail Line2Ds, and init as empty arrays ----
    trail_length = 10  # number of past frames to show
    trails = []
    for _ in range(num_tracks):
        line, = ax.plot([], [], lw=1, alpha=0.7)
        trails.append(line)

    def update(frame_idx):
        pts   = []
        cols  = []
        # start with empty arrays for every track’s trail
        trail_segments = [np.empty((0,2)) for _ in range(num_tracks)]

        for i, pm in enumerate(track_pos_maps):
            if frame_idx in pm:
                # current position
                pts.append(pm[frame_idx])
                cols.append(colors_by_idx[i])

                # collect past positions up to trail_length
                past = []
                for dt in range(trail_length):
                    t = frame_idx - dt
                    if t in pm:
                        past.append(pm[t])
                if past:
                    trail_segments[i] = np.array(past)

        # update scatter
        if pts:
            pts_arr = np.array(pts)
            scat.set_offsets(pts_arr)
            scat.set_color(cols)
        else:
            scat.set_offsets([])

        # update trails
        for i, line in enumerate(trails):
            seg = trail_segments[i]
            if seg.shape[0] > 0:
                line.set_data(seg[:,0], seg[:,1])
                line.set_color(colors_by_idx[i])
            else:
                line.set_data([], [])

        ax.set_xlabel(f"Frame {frame_idx+1}/{num_frames}")
        return (scat, *trails)

    # ---- animate & save MP4 ----
    ani = FuncAnimation(
        fig, update,
        frames=num_frames,
        interval=50,    # ~20 Hz
        blit=True
    )
    out_file = OUT_DIR / f"colored_val_{sid}.mp4"
    ani.save(out_file, fps=20, dpi=150)
    plt.close(fig)
    print(f"[✓] Wrote {out_file}")
