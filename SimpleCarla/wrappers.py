
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
            'camera': spaces.Box(low=0.0, high=1.0, shape=(84, 84, 3), dtype=np.float32),
            'lidar': spaces.Box(low=0.0, high=1.0, shape=(32,), dtype=np.float32),
            'state': spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        })
        
    def observation(self, obs):
        # 1. Camera (Minimap)
        minimap = obs.get('minimap', np.zeros((84,84,3), dtype=np.uint8))
        camera_norm = minimap.astype(np.float32) / 255.0
        
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
        
        dist_right = float(obs.get('dist_to_right_boundary', [10.0])[0])
        dist_right_norm = np.clip(dist_right / 3.5, 0.0, 1.0)
        
        lead_dist = float(obs.get('distance_to_lead', [100.0])[0])
        lead_dist_norm = lead_dist / 50.0
        
        traffic_light = float(obs.get('traffic_light_state', [0.0])[0])  # Already 0 or 1
        
        # Reserved slots
        reserved1 = 0.0
        reserved2 = 0.0
        
        state_vec = np.array([
            vel_norm, steer, lat_norm, head_norm,
            dist_left_norm, dist_right_norm, lead_dist_norm, traffic_light,
            reserved1, reserved2
        ], dtype=np.float32)
        
        return {
            'camera': camera_norm,
            'lidar': lidar_norm,
            'state': state_vec
        }
