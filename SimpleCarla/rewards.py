import numpy as np

class RewardSignal:
    def __init__(self):
        self.max_speed = 30.0 # m/s
        self.max_episode_duration = 30.0 # seconds
        self.current_episode_duration = 0.0

    def reset(self):
        self.current_episode_duration = 0.0

    def compute(self, info, dt):
        """
        Computes the reward step.
        info dict must contain:
        - speed: float (m/s)
        - is_collision: bool
        - is_off_road: bool
        - lane_type: str ('driving', 'shoulder', 'oncoming', 'sidewalk')
        - is_red_light: bool
        - distance_to_lead: float (meters)
        - lateral_error: float
        """
        
        self.current_episode_duration += dt
        
        speed = info.get('speed', 0.0)
        is_collision = info.get('is_collision', False)
        is_off_road = info.get('is_off_road', False)
        lane_type = info.get('lane_type', 'none')
        is_red_light = info.get('is_red_light', False)
        distance_to_lead = info.get('distance_to_lead', 100.0)
        
        total_reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Base Reward (Speed)
        # Using soft clip for speed reward? User asked for +1.0 * (speed / max_speed).
        if speed > 0:
            total_reward += 1.0 * (min(speed, self.max_speed) / self.max_speed)
            
        # 2. Idle Penalty ("No Reason" Check)
        # IF speed < 0.1 AND NOT is_red_light AND distance_to_lead > 5.0
        if speed < 0.1 and not is_red_light and distance_to_lead > 5.0:
            total_reward -= 0.5
            
        # 3. Lane Violation
        if lane_type == 'oncoming':
            total_reward -= 0.5
        elif lane_type == 'shoulder' or lane_type == 'sidewalk': # Assuming sidewalk is bad too
            total_reward -= 0.5
            
        # 4. Terminal Failure
        if is_collision:
            total_reward -= 100.0
            terminated = True # User said "Terminal Failure", implies done.
            # Wait, User said "Truncate" in text: "Terminal Failure (Truncate): IF is_collision...". 
            # Usually Collision is Terminated (natural end). Truncated is time limit.
            # I will allow Terminated here as it's a failure state.
            
        if is_off_road and not is_collision: # Avoid double counting if offroad causes collision
            total_reward -= 100.0
            terminated = True
            
        # 5. Time Limit (Truncation)
        if self.current_episode_duration > self.max_episode_duration:
            truncated = True
            
        return total_reward, terminated, truncated
