"""
Drone Configuration Template
Modify this file to set your specific drone positions for each location.
"""

# Configuration for drone simulation
DRONE_CONFIG = {
    # FOV radius in pixels (adjust based on your needs)
    "fov_radius": 180,
    
    # Drone positions for each location
    # Format: "location_name": [(x1, y1), (x2, y2), ...]
    "drone_positions": {
        
        # Bookstore location
        "bookstore": [
        (250, 250),
        (1000, 300),
        (1000, 750),
        (250, 800)
        ],
        
        # Coupa location
        "coupa": [
        (250, 250),
        (900, 300),
        (950, 750),
        (300, 800),
        (1500, 300),
        (1500, 750)
        ],
        
        # Death Circle location
        "deathCircle": [
        (650, 350),
        (1050, 1000),
        (650, 1500),
        (350, 1000),
        ],
        
        # Gates location
        "gates": [
        (650, 350),
        (1050, 1000),
        (650, 1500),
        (350, 1025),
        ],
        
        # Hyang location
        "hyang": [
        (700, 350),
        (1050, 1025),
        (650, 1500),
        (350, 1025),
        ],
        
        # Little location
        "little": [
        (700, 350),
        (1050, 1025),
        (650, 1500),
        (350, 1025),
        ],
        
        # Nexus location
        "nexus": [
        (550, 500),
        (300, 950),   # drone2
        (800, 950),   # drone3
        (550, 1400)
        ],
        
        # Quad location
        "quad": [
        (500, 250),
        (300, 800),   # drone2
        (1150, 750),   # drone3
        (1500, 300)
        ],
    }
}

# Additional settings
SETTINGS = {
    "dataset_path": "stanford_data/archive",
    "output_path": "new_data",
    "visualization_output": "visualizations",
    "save_visualizations": True,
    "check_overlaps": True,
}

def get_config():
    """Return the drone configuration"""
    return DRONE_CONFIG

def get_settings():
    """Return the settings"""
    return SETTINGS

def validate_config():
    """Validate the drone configuration"""
    from drone_simulator import DroneSimulator
    
    simulator = DroneSimulator(SETTINGS["dataset_path"])
    fov_radius = DRONE_CONFIG["fov_radius"]
    
    print("Validating drone configuration...")
    
    for location, positions in DRONE_CONFIG["drone_positions"].items():
        print(f"\nLocation: {location}")
        print(f"Number of drones: {len(positions)}")
        
        # Check for overlaps
        valid = simulator._validate_drone_positions(positions, fov_radius)
        if valid:
            print("✓ No overlapping FOVs detected")
        else:
            print("✗ WARNING: Overlapping FOVs detected!")
            
            # Show which drones overlap
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    if simulator._check_fov_overlap(positions[i], positions[j], fov_radius):
                        print(f"  Drones {i+1} and {j+1} have overlapping FOVs")
        
        # Show positions
        for i, pos in enumerate(positions):
            print(f"  Drone {i+1}: ({pos[0]}, {pos[1]})")

if __name__ == "__main__":
    validate_config()