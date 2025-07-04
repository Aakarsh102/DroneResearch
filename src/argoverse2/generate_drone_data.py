# # enhanced_drone_simulation.py

# import numpy as np
# import pandas as pd
# from pathlib import Path
# import json
# from typing import Dict, List, Tuple, Optional
# from dataclasses import dataclass, asdict

# # Patch for AV2's typing
# if not hasattr(np, "bool"):
#     np.bool = bool

# from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
# from av2.map.map_api import ArgoverseStaticMap


# @dataclass
# class DroneConfig:
#     """Configuration for a drone"""
#     drone_id: str
#     center_x: float
#     center_y: float
#     radius: float
#     fov_angle: float = 360.0  # Full circle by default
#     facing_direction: float = 0.0  # Direction drone is facing (degrees)
#     height: float = 50.0  # Drone height (for future 3D calculations)
    

# @dataclass
# class AgentObservation:
#     """Single agent observation by a drone"""
#     frame: int
#     timestamp_ns: int
#     drone_id: str
#     track_id: str
#     x: float
#     y: float
#     distance: float
#     is_focal: bool
#     agent_type: str


# class DroneSimulator:
#     """Enhanced drone simulation with better FOV modeling and analytics"""
    
#     def __init__(self, scenario_root: Path, output_dir: Path):
#         self.scenario_root = Path(scenario_root)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(exist_ok=True)
        
#     def calculate_scene_bounds(self, track_data: List[Dict]) -> Tuple[float, float, float, float]:
#         """Calculate scene boundaries with padding"""
#         all_pts = [pt for td in track_data for pt in td['pos_map'].values()]
#         if not all_pts:
#             return 0, 100, 0, 100
            
#         xs, ys = zip(*all_pts)
#         pad = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.1  # 10% padding
#         return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    
#     def create_optimal_drone_positions(self, xmin: float, xmax: float, 
#                                      ymin: float, ymax: float) -> List[DroneConfig]:
#         """Create 5 optimally positioned drones with non-overlapping FOVs"""
        
#         # Calculate scene dimensions
#         width = xmax - xmin
#         height = ymax - ymin
#         cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        
#         # Set radius to cover good area without overlap
#         # Using smaller radius to ensure no overlap
#         base_radius = min(width, height) / 8.0
        
#         # Create drone configurations
#         drones = [
#             # Center drone - smaller radius to avoid overlap
#             DroneConfig("center", cx, cy, base_radius * 0.8),
            
#             # Four corner drones positioned to avoid overlap
#             DroneConfig("left", cx - width * 0.3, cy, base_radius),
#             DroneConfig("right", cx + width * 0.3, cy, base_radius),
#             DroneConfig("top", cx, cy + height * 0.3, base_radius),
#             DroneConfig("bottom", cx, cy - height * 0.3, base_radius)
#         ]
        
#         # Verify no overlaps
#         self._verify_no_overlaps(drones)
        
#         return drones
    
#     def _verify_no_overlaps(self, drones: List[DroneConfig]) -> None:
#         """Verify that drone FOVs don't overlap"""
#         for i, drone1 in enumerate(drones):
#             for j, drone2 in enumerate(drones[i+1:], i+1):
#                 distance = np.sqrt((drone1.center_x - drone2.center_x)**2 + 
#                                  (drone1.center_y - drone2.center_y)**2)
#                 min_distance = drone1.radius + drone2.radius
                
#                 if distance < min_distance:
#                     print(f"WARNING: Drones {drone1.drone_id} and {drone2.drone_id} "
#                           f"FOVs may overlap! Distance: {distance:.2f}, "
#                           f"Required: {min_distance:.2f}")
    
#     def is_agent_in_fov(self, agent_x: float, agent_y: float, 
#                        drone: DroneConfig) -> Tuple[bool, float]:
#         """Check if agent is within drone's field of view"""
        
#         # Calculate distance
#         dx = agent_x - drone.center_x
#         dy = agent_y - drone.center_y
#         distance = np.sqrt(dx**2 + dy**2)
        
#         # Check if within radius (360° FOV)
#         return distance <= drone.radius, distance
    
#     def simulate_drone_observations(self, track_data: List[Dict], 
#                                   timestamps: List[int], 
#                                   drone: DroneConfig) -> List[AgentObservation]:
#         """Simulate observations for a single drone across all frames"""
        
#         observations = []
        
#         for frame_idx, timestamp in enumerate(timestamps):
#             for td in track_data:
#                 if frame_idx in td['pos_map']:
#                     x, y = td['pos_map'][frame_idx]
                    
#                     # Check if agent is in FOV
#                     in_fov, distance = self.is_agent_in_fov(x, y, drone)
                    
#                     if in_fov:
#                         obs = AgentObservation(
#                             frame=frame_idx,
#                             timestamp_ns=timestamp,
#                             drone_id=drone.drone_id,
#                             track_id=td['track_id'],
#                             x=x,
#                             y=y,
#                             distance=distance,
#                             is_focal=td['is_focal'],
#                             agent_type=td['agent_type']
#                         )
                        
#                         observations.append(obs)
        
#         return observations
    
#     def generate_analytics(self, all_observations: Dict[str, List[AgentObservation]], 
#                           scenario_id: str) -> Dict:
#         """Generate analytics for the drone observations"""
        
#         analytics = {
#             'scenario_id': scenario_id,
#             'total_observations': sum(len(obs) for obs in all_observations.values()),
#             'drones': {}
#         }
        
#         for drone_id, observations in all_observations.items():
#             if not observations:
#                 continue
                
#             # Basic stats
#             unique_agents = len(set(obs.track_id for obs in observations))
#             focal_observations = sum(1 for obs in observations if obs.is_focal)
            
#             # Distance stats
#             distances = [obs.distance for obs in observations]
            
#             # Coverage stats
#             frames_with_observations = len(set(obs.frame for obs in observations))
            
#             # Agent type distribution
#             agent_types = {}
#             for obs in observations:
#                 agent_types[obs.agent_type] = agent_types.get(obs.agent_type, 0) + 1
            
#             analytics['drones'][drone_id] = {
#                 'total_observations': len(observations),
#                 'unique_agents_seen': unique_agents,
#                 'focal_agent_observations': focal_observations,
#                 'frames_with_data': frames_with_observations,
#                 'agent_type_distribution': agent_types,
#                 'distance_stats': {
#                     'mean': np.mean(distances),
#                     'std': np.std(distances),
#                     'min': np.min(distances),
#                     'max': np.max(distances)
#                 }
#             }
        
#         return analytics
    
#     def process_scenario(self, scenario_dir: Path) -> Optional[Dict]:
#         """Process a single scenario"""
        
#         sid = scenario_dir.name
#         pq_path = scenario_dir / f"scenario_{sid}.parquet"
#         map_path = scenario_dir / f"log_map_archive_{sid}.json"
        
#         if not pq_path.exists() or not map_path.exists():
#             print(f"[!] Missing files for {sid}, skipping.")
#             return None
        
#         try:
#             # Load scenario
#             scenario = load_argoverse_scenario_parquet(pq_path)
#             static_map = ArgoverseStaticMap.from_json(map_path)
            
#             # Build track data with correct agent types
#             track_data = []
#             for tr in scenario.tracks:
#                 # Get the correct object type from Argoverse2
#                 agent_type = str(tr.object_type).split('.')[-1] if hasattr(tr, 'object_type') else 'UNKNOWN'
                
#                 track_data.append({
#                     'track_id': tr.track_id,
#                     'pos_map': {st.timestep: (st.position[0], st.position[1]) for st in tr.object_states},
#                     'is_focal': (tr.track_id == scenario.focal_track_id),
#                     'agent_type': agent_type
#                 })
            
#             # Calculate bounds and create drone positions
#             xmin, xmax, ymin, ymax = self.calculate_scene_bounds(track_data)
#             drones = self.create_optimal_drone_positions(xmin, xmax, ymin, ymax)
            
#             # Simulate observations for each drone
#             all_observations = {}
            
#             for drone in drones:
#                 observations = self.simulate_drone_observations(
#                     track_data, list(scenario.timestamps_ns), drone
#                 )
#                 all_observations[drone.drone_id] = observations
                
#                 # Save individual drone data
#                 if observations:
#                     df = pd.DataFrame([asdict(obs) for obs in observations])
#                     csv_path = self.output_dir / f'{drone.drone_id}_observations_{sid}.csv'
#                     df.to_csv(csv_path, index=False)
#                     print(f"[✓] Saved {drone.drone_id} data: {len(observations)} observations")
            
#             # Generate and save analytics
#             analytics = self.generate_analytics(all_observations, sid)
            
#             analytics_path = self.output_dir / f'analytics_{sid}.json'
#             with open(analytics_path, 'w') as f:
#                 json.dump(analytics, f, indent=2, default=str)
            
#             print(f"[✓] Processed scenario {sid}: {analytics['total_observations']} total observations")
            
#             return analytics
            
#         except Exception as e:
#             print(f"[!] Error processing {sid}: {e}")
#             return None
    
#     def run_simulation(self) -> None:
#         """Run simulation on all scenarios"""
        
#         print("Starting enhanced drone simulation...")
        
#         all_analytics = []
        
#         for scenario_dir in sorted(self.scenario_root.iterdir()):
#             if not scenario_dir.is_dir():
#                 continue
                
#             analytics = self.process_scenario(scenario_dir)
#             if analytics:
#                 all_analytics.append(analytics)
        
#         # Save summary analytics
#         summary_path = self.output_dir / 'simulation_summary.json'
#         with open(summary_path, 'w') as f:
#             json.dump(all_analytics, f, indent=2, default=str)
        
#         print(f"\n[✓] Simulation complete! Processed {len(all_analytics)} scenarios")
#         print(f"[✓] Results saved to {self.output_dir}")


# # ────────────────────────────────────────────────────────────────────────────
# # MAIN EXECUTION
# # ────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     # Configuration
#     SCENARIO_ROOT = Path("argoverse2_data/val")
#     OUT_DIR = Path("src/argoverse2/enhanced_drone_datasets")
    
#     # Create and run simulator
#     simulator = DroneSimulator(SCENARIO_ROOT, OUT_DIR)
#     simulator.run_simulation()

# enhanced_drone_simulation.py

# enhanced_drone_simulation.py

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

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
    x_local: float  # Local x coordinate relative to drone
    y_local: float  # Local y coordinate relative to drone
    x_global: float  # Global x coordinate (for reference)
    y_global: float  # Global y coordinate (for reference)
    distance: float
    is_focal: bool
    agent_type: str


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
            for j, drone2 in enumerate(drones[i+1:], i+1):
                distance = np.sqrt((drone1.center_x - drone2.center_x)**2 + 
                                 (drone1.center_y - drone2.center_y)**2)
                min_distance = drone1.radius + drone2.radius
                
                if distance < min_distance:
                    print(f"WARNING: Drones {drone1.drone_id} and {drone2.drone_id} "
                          f"FOVs may overlap! Distance: {distance:.2f}, "
                          f"Required: {min_distance:.2f}")
    
    def global_to_local_coordinates(self, global_x: float, global_y: float, 
                                  drone: DroneConfig) -> Tuple[float, float]:
        """Convert global coordinates to drone's local coordinate system (BEV)"""
        # Simple translation: drone is at origin (0,0) in its local system
        local_x = global_x - drone.center_x
        local_y = global_y - drone.center_y
        return local_x, local_y
    
    def is_agent_in_fov(self, agent_x: float, agent_y: float, 
                       drone: DroneConfig) -> Tuple[bool, float]:
        """Check if agent is within drone's field of view"""
        
        # Calculate distance
        dx = agent_x - drone.center_x
        dy = agent_y - drone.center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Check if within radius (360° FOV)
        return distance <= drone.radius, distance
    
    def simulate_drone_observations(self, track_data: List[Dict], 
                                  timestamps: List[int], 
                                  drone: DroneConfig) -> List[AgentObservation]:
        """Simulate observations for a single drone across all frames"""
        
        observations = []
        
        for frame_idx, timestamp in enumerate(timestamps):
            for td in track_data:
                if frame_idx in td['pos_map']:
                    global_x, global_y = td['pos_map'][frame_idx]
                    
                    # Check if agent is in FOV
                    in_fov, distance = self.is_agent_in_fov(global_x, global_y, drone)
                    
                    if in_fov:
                        # Convert to local coordinates
                        local_x, local_y = self.global_to_local_coordinates(
                            global_x, global_y, drone
                        )
                        
                        obs = AgentObservation(
                            frame=frame_idx,
                            timestamp_ns=timestamp,
                            drone_id=drone.drone_id,
                            track_id=td['track_id'],
                            x_local=local_x,
                            y_local=local_y,
                            x_global=global_x,
                            y_global=global_y,
                            distance=distance,
                            is_focal=td['is_focal'],
                            agent_type=td['agent_type']
                        )
                        
                        observations.append(obs)
        
        return observations
    
    def create_empty_csv_with_headers(self, csv_path: Path) -> None:
        """Create an empty CSV file with proper headers"""
        # Create a DataFrame with the correct columns but no data
        empty_df = pd.DataFrame(columns=[
            'frame', 'timestamp_ns', 'drone_id', 'track_id', 
            'x_local', 'y_local', 'x_global', 'y_global', 
            'distance', 'is_focal', 'agent_type'
        ])
        empty_df.to_csv(csv_path, index=False)
    
    def generate_analytics(self, all_observations: Dict[str, List[AgentObservation]], 
                          scenario_id: str, drones: List[DroneConfig]) -> Dict:
        """Generate analytics for the drone observations"""
        
        analytics = {
            'scenario_id': scenario_id,
            'total_observations': sum(len(obs) for obs in all_observations.values()),
            'drone_configs': {
                drone.drone_id: {
                    'center_x_global': drone.center_x,
                    'center_y_global': drone.center_y,
                    'radius': drone.radius,
                    'facing_direction': drone.facing_direction,
                    'height': drone.height
                }
                for drone in drones
            },
            'drones': {}
        }
        
        for drone_id, observations in all_observations.items():
            if not observations:
                # Still add entry for empty drones
                analytics['drones'][drone_id] = {
                    'total_observations': 0,
                    'unique_agents_seen': 0,
                    'focal_agent_observations': 0,
                    'frames_with_data': 0,
                    'agent_type_distribution': {},
                }
                continue
                
            # Basic stats
            unique_agents = len(set(obs.track_id for obs in observations))
            focal_observations = sum(1 for obs in observations if obs.is_focal)
            
            # Distance stats
            distances = [obs.distance for obs in observations]
            
            # Local coordinate stats
            local_x_coords = [obs.x_local for obs in observations]
            local_y_coords = [obs.y_local for obs in observations]
            
            # Coverage stats
            frames_with_observations = len(set(obs.frame for obs in observations))
            
            # Agent type distribution
            agent_types = {}
            for obs in observations:
                agent_types[obs.agent_type] = agent_types.get(obs.agent_type, 0) + 1
            
            analytics['drones'][drone_id] = {
                'total_observations': len(observations),
                'unique_agents_seen': unique_agents,
                'focal_agent_observations': focal_observations,
                'frames_with_data': frames_with_observations,
                'agent_type_distribution': agent_types,
                'distance_stats': {
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'min': np.min(distances),
                    'max': np.max(distances)
                },
                'local_coordinate_bounds': {
                    'x_min': np.min(local_x_coords),
                    'x_max': np.max(local_x_coords),
                    'y_min': np.min(local_y_coords),
                    'y_max': np.max(local_y_coords)
                }
            }
        
        return analytics
    
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
            
            # Build track data with correct agent types
            track_data = []
            for tr in scenario.tracks:
                # Get the correct object type from Argoverse2
                agent_type = str(tr.object_type).split('.')[-1] if hasattr(tr, 'object_type') else 'UNKNOWN'
                
                track_data.append({
                    'track_id': tr.track_id,
                    'pos_map': {st.timestep: (st.position[0], st.position[1]) for st in tr.object_states},
                    'is_focal': (tr.track_id == scenario.focal_track_id),
                    'agent_type': agent_type
                })
            
            # Calculate bounds and create drone positions
            xmin, xmax, ymin, ymax = self.calculate_scene_bounds(track_data)
            drones = self.create_optimal_drone_positions(xmin, xmax, ymin, ymax)
            
            # Simulate observations for each drone
            all_observations = {}
            
            for drone in drones:
                observations = self.simulate_drone_observations(
                    track_data, list(scenario.timestamps_ns), drone
                )
                all_observations[drone.drone_id] = observations
                
                # Save individual drone data
                csv_path = self.output_dir / f'{drone.drone_id}_observations_{sid}.csv'
                
                if observations:
                    # Create DataFrame from observations
                    df = pd.DataFrame([asdict(obs) for obs in observations])
                    df.to_csv(csv_path, index=False)
                    print(f"[✓] Saved {drone.drone_id} data: {len(observations)} observations")
                else:
                    # Create empty CSV with proper headers
                    self.create_empty_csv_with_headers(csv_path)
                    print(f"[!] {drone.drone_id} captured 0 observations (empty CSV with headers saved)")
            
            # Generate and save analytics (includes drone configs now)
            analytics = self.generate_analytics(all_observations, sid, drones)
            
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
        
        print("Starting enhanced drone simulation with local coordinates...")
        
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
    SCENARIO_ROOT = Path("argoverse2_data/val")
    OUT_DIR = Path("src/argoverse2/enhanced_drone_datasets_final")
    
    # Create and run simulator
    simulator = DroneSimulator(SCENARIO_ROOT, OUT_DIR)
    simulator.run_simulation()