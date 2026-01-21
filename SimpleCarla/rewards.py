import math

class RewardSignal:
    def __init__(self):
        self.target_speed = 20.0
        self.max_time = 30.0  # seconds for success
        # For HUD display compatibility
        self.max_episode_duration = 30.0
        self.current_episode_duration = 0.0
    
    def reset(self):
        """Reset episode duration tracking"""
        self.current_episode_duration = 0.0

    def compute(self, info, dt):
        """
        Compute reward based on Survival Formula.
        
        info dict keys:
        - speed (m/s)
        - is_collision (bool)
        - is_off_road (bool)
        - lane_type (str)
        - is_red_light (bool)
        - lateral_error (m)
        - min_lidar_dist (m)
        - current_time (s)
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # Track episode time
        self.current_episode_duration += dt
        
        # 1. Speed Reward (0.0 to 2.0) - INCREASED from 1.0
        speed = info.get('speed', 0.0)
        
        # Scale to 0-2.0 (doubled to overcome penalties)
        r_speed = 2.0 * (speed / self.target_speed)
        
        # Penalty for speeding (Target + 5)
        if speed > (self.target_speed + 5.0):
             r_speed *= 0.5
        
        reward += r_speed
        
        # 2. Centering Penalty (Moderate) - REDUCED from -1.0/m to -0.3/m
        # Only penalize significant lateral errors
        lat_err = abs(info.get('lateral_error', 0.0))
        r_centering = -0.3 * lat_err
        reward += r_centering
        
        # 3. Proximity Penalty (Force Field) - REDUCED threshold from 15m to 8m
        # Only penalize dangerously close following, not normal gaps
        min_dist = info.get('min_lidar_dist', 50.0)
        r_prox = 0.0
        if min_dist < 8.0:
            # (8 - dist) / 8 -> 0 at 8m, 1 at 0m
            # Scale to -0.5 max penalty (reduced from -1.0)
            r_prox = -0.5 * (8.0 - min_dist) / 8.0
        reward += r_prox
        
        # 4. Idle Penalty - UNCHANGED
        # If speed < 0.1 AND !RedLight AND Front Clear
        is_red = info.get('is_red_light', False)
        if speed < 0.1 and not is_red and min_dist > 10.0:
            reward -= 0.5
            
        # 5. Wrong Way / Shoulder - UNCHANGED
        lane_type = info.get('lane_type', 'none')
        if lane_type != 'driving':
            reward -= 1.0
            
        # 6. Terminal States
        if info.get('is_collision', False):
            reward = -50.0
            terminated = True
            
        if info.get('is_off_road', False):
             reward = -50.0
             terminated = True  # Stop immediately
        
        # Success (Survival Time)
        current_time = info.get('current_time', 0.0)
        if current_time > self.max_time:
             reward += 50.0
             truncated = True

        return reward, terminated, truncated
