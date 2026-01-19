import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CarlaObservationWrapper(gym.ObservationWrapper):
    """
    Normalizes the observation space for PPO training.
    Output is a Dict space:
    - 'lidar': (N,) float32, normalized [0, 1] (dist / max_range)
    - 'state': (2,) float32, [velocity/max_speed, steering]
    - 'navigation': (2,) float32, [lateral_error, heading_error] normalized roughly to [-1, 1]
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Constants for Normalization
        self.MAX_LIDAR_RANGE = 50.0
        self.MAX_SPEED = 30.0 # m/s
        self.full_observation_space = env.observation_space # Save original
        
        # Define New Space
        # Assuming original env returns a dict with 'lidar', 'state', etc.
        # If original env returns image, we are changing it here entirely as per request.
        
        # We need to know the Lidar size. Let's assume N=360 or similar.
        # We will inspect env.lidar_rays if available, else default.
        lidar_dim = getattr(env, 'lidar_rays', 360) 
        
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0.0, high=1.0, shape=(lidar_dim,), dtype=np.float32),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32), # [vel, steer]
            'navigation': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # [lat_err, head_err]
        })
        
    def observation(self, obs):
        # 'obs' here is what the Env returns.
        # The Env likely returns a simpler Dict or Raw values.
        # We expect the Env to be updated to return a Dict containing 'raw_lidar', 'speed', 'steering', 'lat_err', 'head_err'.
        
        # Lidar
        raw_lidar = obs.get('lidar', np.zeros(self.observation_space['lidar'].shape))
        norm_lidar = np.clip(raw_lidar / self.MAX_LIDAR_RANGE, 0.0, 1.0).astype(np.float32)
        
        # State
        speed = obs.get('speed', 0.0)
        steering = obs.get('steering', 0.0)
        norm_speed = float(speed / self.MAX_SPEED)
        norm_state = np.array([norm_speed, steering], dtype=np.float32)
        
        # Navigation
        lat_err = obs.get('lateral_error', 0.0)
        head_err = obs.get('heading_error', 0.0)
        
        # Normalize Lateral Error (Clamp to reasonable range, e.g., +/- 5m lane width?)
        # Let's assume input is meters. We clamp to [-1, 1] conceptually for RL stability, 
        # but let's just scale by 2.0m for now? Or just clip?
        # User said "normalized to [-1, 1]".
        # We'll map +/- 4 meters to +/- 1.0.
        norm_lat = np.clip(lat_err / 4.0, -1.0, 1.0)
        
        # Heading Error: Range [-pi, pi]. Normalize by pi.
        norm_head = np.clip(head_err / np.pi, -1.0, 1.0)
        
        norm_nav = np.array([norm_lat, norm_head], dtype=np.float32)
        
        return {
            'lidar': norm_lidar,
            'state': norm_state,
            'navigation': norm_nav
        }
