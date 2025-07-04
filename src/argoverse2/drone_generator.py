# enhanced_drone_simulation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Patch for AV2's typing
if not hasattr(np, "bool"):
    np.bool = bool

from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap


@dataclass
class DroneConfig:
    """Configuration for a drone"""
    drone_id: str
    center_x: float
    center_y: float
    radius: float
    fov_angle: float = 360.0  # Full circle by default
    facing_direction: float = 0.0  # Direction drone is facing (degrees)
    height: float = 50.0  # Drone height (for future 3D calculations)


@dataclass
class AgentObservation:
    """Single agent observation by a drone"""
    frame: int
    timestamp_ns: int
    drone_id: str
    track_id: str
    x: float
    y: float
    distance: float
    angle_from_drone: float
    is_focal: bool
    agent_type: str
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    heading: float = 0.0


class DroneSimulator:
    """Enhanced drone simulation with better FOV modeling and analytics"""

    def __init__(self, scenario_root: Path, output_dir: Path):
        self.scenario_root = Path(scenario_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def calculate_scene_bounds(self, track_data: List[Dict]) -> Tuple[float, float, float, float]:
        """Calculate scene boundaries with padding"""
        all_pts = [pt for td in track_data for pt in td['pos_map'].values()]
        if not all_pts:
            return 0, 100, 0, 100

        xs, ys = zip(*all_pts)
        pad = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.1  # 10% padding
        return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad

    def create_optimal_drone_positions(self, xmin: float, xmax: float,
                                       ymin: float, ymax: float) -> List[DroneConfig]:
        """Create 5 optimally positioned drones with non-overlapping FOVs"""

        # Calculate scene dimensions
        width = xmax - xmin
        height = ymax - ymin
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

        # Set radius to cover good area without overlap
        # Using smaller radius to ensure no overlap
        base_radius = min(width, height) / 8.0

        # Create drone configurations
        drones = [
            # Center drone - smaller radius to avoid overlap
            DroneConfig("center", cx, cy, base_radius * 0.8),

            # Four corner drones positioned to avoid overlap
            DroneConfig("left", cx - width * 0.3, cy, base_radius),
            DroneConfig("right", cx + width * 0.3, cy, base_radius),
            DroneConfig("top", cx, cy + height * 0.3, base_radius),
            DroneConfig("bottom", cx, cy - height * 0.3, base_radius)
        ]

        # Verify no overlaps
        self._verify_no_overlaps(drones)

        return drones

    def _verify_no_overlaps(self, drones: List[DroneConfig]) -> None:
        """Verify that drone FOVs don't overlap"""
        for i, drone1 in enumerate(drones):
            for j, drone2 in enumerate(drones[i + 1:], i + 1):
                distance = np.sqrt((drone1.center_x - drone2.center_x) ** 2 +
                                   (drone1.center_y - drone2.center_y) ** 2)
                min_distance = drone1.radius + drone2.radius

                if distance <= min_distance:
                    print(f"WARNING: Drones {drone1.drone_id} and {drone2.drone_id} "
                          f"FOVs may overlap! Distance: {distance:.2f}, "
                          f"Required: {min_distance:.2f}")

    def is_agent_in_fov(self, agent_x: float, agent_y: float,
                        drone: DroneConfig) -> Tuple[bool, float, float]:
        """Check if agent is within drone's field of view"""

        # Calculate distance
        dx = agent_x - drone.center_x
        dy = agent_y - drone.center_y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Check if within radius
        if distance > drone.radius:
            return False, distance, 0.0

        # Calculate angle from drone
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360

        # For full 360° FOV, always in view if within radius
        if drone.fov_angle >= 360:
            return True, distance, angle

        # Check if within FOV angle
        # Calculate angle difference from facing direction
        angle_diff = abs(angle - drone.facing_direction)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff <= drone.fov_angle / 2, distance, angle

    def calculate_agent_velocity(self, track_data: Dict, frame_idx: int) -> Tuple[float, float]:
        """Calculate agent velocity between frames"""
        pos_map = track_data['pos_map']

        if frame_idx in pos_map and (frame_idx - 1) in pos_map:
            curr_pos = pos_map[frame_idx]
            prev_pos = pos_map[frame_idx - 1]

            # Assuming 10Hz data (0.1s between frames)
            dt = 0.1
            vx = (curr_pos[0] - prev_pos[0]) / dt
            vy = (curr_pos[1] - prev_pos[1]) / dt

            return vx, vy

        return 0.0, 0.0

    def simulate_drone_observations(self, track_data: List[Dict],
                                    timestamps: List[int],
                                    drone: DroneConfig) -> List[AgentObservation]:
        """Simulate observations for a single drone across all frames"""

        observations = []

        for frame_idx, timestamp in enumerate(timestamps):
            for td in track_data:
                if frame_idx in td['pos_map']:
                    x, y = td['pos_map'][frame_idx]

                    # Check if agent is in FOV
                    in_fov, distance, angle = self.is_agent_in_fov(x, y, drone)

                    if in_fov:
                        # Calculate velocity
                        vx, vy = self.calculate_agent_velocity(td, frame_idx)

                        # Calculate heading
                        heading = np.degrees(np.arctan2(vy, vx)) if (vx != 0 or vy != 0) else 0.0

                        obs = AgentObservation(
                            frame=frame_idx,
                            timestamp_ns=timestamp,
                            drone_id=drone.drone_id,
                            track_id=td['track_id'],
                            x=x,
                            y=y,
                            distance=distance,
                            angle_from_drone=angle,
                            is_focal=td['is_focal'],
                            agent_type=td.get('agent_type', 'unknown'),
                            velocity_x=vx,
                            velocity_y=vy,
                            heading=heading
                        )

                        observations.append(obs)

        return observations

    def generate_analytics(self, all_observations: Dict[str, List[AgentObservation]],
                           scenario_id: str) -> Dict:
        """Generate analytics for the drone observations"""

        analytics = {
            'scenario_id': scenario_id,
            'total_observations': sum(len(obs) for obs in all_observations.values()),
            'drones': {}
        }

        for drone_id, observations in all_observations.items():
            if not observations:
                continue

            # Basic stats
            unique_agents = len(set(obs.track_id for obs in observations))
            focal_observations = sum(1 for obs in observations if obs.is_focal)

            # Distance stats
            distances = [obs.distance for obs in observations]

            # Coverage stats
            frames_with_observations = len(set(obs.frame for obs in observations))

            analytics['drones'][drone_id] = {
                'total_observations': len(observations),
                'unique_agents_seen': unique_agents,
                'focal_agent_observations': focal_observations,
                'frames_with_data': frames_with_observations,
                'distance_stats': {
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'min': np.min(distances),
                    'max': np.max(distances)
                }
            }

        return analytics

    def visualize_drone_setup(self, drones: List[DroneConfig],
                              xmin: float, xmax: float, ymin: float, ymax: float,
                              scenario_id: str) -> None:
        """Create visualization of drone positions and FOVs"""

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, drone in enumerate(drones):
            color = colors[i % len(colors)]

            # Draw FOV circle
            circle = patches.Circle((drone.center_x, drone.center_y),
                                    drone.radius,
                                    fill=False,
                                    edgecolor=color,
                                    linewidth=2,
                                    alpha=0.7)
            ax.add_patch(circle)

            # Draw drone position
            ax.scatter(drone.center_x, drone.center_y,
                       c=color, s=100, marker='^',
                       label=f'Drone {drone.drone_id}')

            # Add drone label
            ax.annotate(drone.drone_id,
                        (drone.center_x, drone.center_y),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax.legend()
        ax.set_title(f'Drone FOV Setup - Scenario {scenario_id}')

        # Save visualization
        vis_path = self.output_dir / f'drone_setup_{scenario_id}.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[✓] Drone setup visualization saved to {vis_path}")

    def process_scenario(self, scenario_dir: Path) -> Optional[Dict]:
        """Process a single scenario"""

        sid = scenario_dir.name
        pq_path = scenario_dir / f"scenario_{sid}.parquet"
        map_path = scenario_dir / f"log_map_archive_{sid}.json"

        if not pq_path.exists() or not map_path.exists():
            print(f"[!] Missing files for {sid}, skipping.")
            return None

        try:
            # Load scenario
            scenario = load_argoverse_scenario_parquet(pq_path)
            static_map = ArgoverseStaticMap.from_json(map_path)

            # Build track data
            track_data = []
            for tr in scenario.tracks:
                track_data.append({
                    'track_id': tr.track_id,
                    'pos_map': {st.timestep: st.position for st in tr.object_states},
                    'is_focal': (tr.track_id == scenario.focal_track_id),
                    'agent_type': getattr(tr, 'object_type', 'unknown')
                })

            # Calculate bounds and create drone positions
            xmin, xmax, ymin, ymax = self.calculate_scene_bounds(track_data)
            drones = self.create_optimal_drone_positions(xmin, xmax, ymin, ymax)

            # Create visualization
            self.visualize_drone_setup(drones, xmin, xmax, ymin, ymax, sid)

            # Simulate observations for each drone
            all_observations = {}

            for drone in drones:
                observations = self.simulate_drone_observations(
                    track_data, list(scenario.timestamps_ns), drone
                )
                all_observations[drone.drone_id] = observations

                # Save individual drone data
                if observations:
                    df = pd.DataFrame([asdict(obs) for obs in observations])
                    csv_path = self.output_dir / f'{drone.drone_id}_observations_{sid}.csv'
                    df.to_csv(csv_path, index=False)
                    print(f"[✓] Saved {drone.drone_id} data: {len(observations)} observations")

            # Generate and save analytics
            analytics = self.generate_analytics(all_observations, sid)

            analytics_path = self.output_dir / f'analytics_{sid}.json'
            with open(analytics_path, 'w') as f:
                json.dump(analytics, f, indent=2, default=str)

            print(f"[✓] Processed scenario {sid}: {analytics['total_observations']} total observations")

            return analytics

        except Exception as e:
            print(f"[!] Error processing {sid}: {e}")
            return None

    def run_simulation(self) -> None:
        """Run simulation on all scenarios"""

        print("Starting enhanced drone simulation...")

        all_analytics = []

        for scenario_dir in sorted(self.scenario_root.iterdir()):
            if not scenario_dir.is_dir():
                continue

            analytics = self.process_scenario(scenario_dir)
            if analytics:
                all_analytics.append(analytics)

        # Save summary analytics
        summary_path = self.output_dir / 'simulation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_analytics, f, indent=2, default=str)

        print(f"\n[✓] Simulation complete! Processed {len(all_analytics)} scenarios")
        print(f"[✓] Results saved to {self.output_dir}")


# ────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Configuration
    SCENARIO_ROOT = Path("../../argoverse2_data/val")
    OUT_DIR = Path("enhanced_drone_datasets")

    # Create and run simulator
    simulator = DroneSimulator(SCENARIO_ROOT, OUT_DIR)
    simulator.run_simulation()