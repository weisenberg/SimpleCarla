
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CarlaObservationWrapper(gym.ObservationWrapper):
    """
    Normalizes observations for stable PPO training.
    
    Inputs:
    - minimap: (84, 84, 3) uint8 [0-255]
    - lidar: (32,) float32 [0-50]
    - speed, steering, lateral_error, heading_error
    - NEW: dist_to_left_boundary, dist_to_right_boundary, distance_to_lead, traffic_light_state
    
    Outputs (Normalized):
    - camera: (84, 84, 3) float32 [0, 1]
    - lidar: (32,) float32 [0, 1]
    - state: (10,) float32 [-1, 1]
        [0]: speed / 30.0
        [1]: steering
        [2]: lateral_error / 3.5
        [3]: heading_error / pi
        [4]: dist_to_left_boundary / 3.5
        [5]: dist_to_right_boundary / 3.5
        [6]: distance_to_lead / 50.0
        [7]: traffic_light_state (0 or 1, already normalized)
        [8-9]: Reserved for future sensors
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Define New Spaces
        self.observation_space = spaces.Dict({
            # 'camera' REMOVED for speed
            'lidar': spaces.Box(low=0.0, high=1.0, shape=(32,), dtype=np.float32),
            'state': spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        })
        
    def observation(self, obs):
        # 1. Camera (Minimap) - REMOVED
        # minimap = obs.get('minimap', np.zeros((84,84,3), dtype=np.uint8))
        # camera_norm = minimap.astype(np.float32) / 255.0
        
        # 2. Lidar
        lidar_raw = obs.get('lidar', np.zeros((32,), dtype=np.float32))
        lidar_norm = lidar_raw / 50.0 # Max Range
        
        # 3. State Vector (10 dimensions now)
        speed = float(obs.get('speed', [0.0])[0])
        vel_norm = speed / 30.0
        
        steer = float(obs.get('steering', [0.0])[0])
        
        lat_err = float(obs.get('lateral_error', [0.0])[0])
        lat_norm = np.clip(lat_err / 3.5, -1.0, 1.0)
        
        head_err = float(obs.get('heading_error', [0.0])[0])
        head_norm = head_err / 3.14159
        
        # NEW SENSORS
        dist_left = float(obs.get('dist_to_left_boundary', [10.0])[0])
        dist_left_norm = np.clip(dist_left / 3.5, 0.0, 1.0)
        
        dist_left = float(obs.get('dist_to_left_boundary', [0.0])[0])
        dist_right = float(obs.get('dist_to_right_boundary', [0.0])[0])
        dist_lead = float(obs.get('distance_to_lead', [50.0])[0])
        light_state = float(obs.get('traffic_light_state', [0.0])[0])

        # New: Navigation (Target Point 2D)
        tgt = obs.get('target_point', [0.0, 0.0])
        tgt_x = float(tgt[0]) / 30.0 # Normalize approx
        tgt_y = float(tgt[1]) / 30.0 

        # State Vector (12 dimensions)
        state = np.array([
            vel_norm,      # 0
            steer,         # 1
            lat_norm,      # 2
            head_norm,     # 3
            dist_left/10.0, # 4
            dist_right/10.0,# 5
            dist_lead/50.0, # 6
            light_state,    # 7
            tgt_x,          # 8
            tgt_y,          # 9
            0.0,            # 10 (Reserved)
            0.0             # 11 (Reserved)
        ], dtype=np.float32)
        
        return {
            'lidar': lidar_norm,
            'state': state
        }
