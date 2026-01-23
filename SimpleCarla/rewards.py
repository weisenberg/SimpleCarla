import math

class RewardSignal:
    def __init__(self):
        self.target_speed = 20.0
        self.max_time = 30.0  # seconds for success
        # For HUD display compatibility
        self.max_episode_duration = 30.0
        self.current_episode_duration = 0.0
        self.wrong_way_counter = 0 # Track consecutive wrong-way steps
    
    def reset(self):
        """Reset episode duration and counters"""
        self.current_episode_duration = 0.0
        self.wrong_way_counter = 0

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
        
        # 1. Progress Reward (Alignment-Based Speed)
        # Prevents "donuts" by rewarding speed ONLY in the correct direction
        speed = info.get('speed', 0.0)
        heading_error = info.get('heading_error', 0.0)
        min_dist = info.get('min_lidar_dist', 50.0)
        
        # Alignment Factor: 1.0 = Perfect, 0.0 = Perpendicular, -1.0 = Backwards
        alignment = math.cos(heading_error)
        
        # Safe Speed Logic
        if min_dist < 15.0:
            r_progress = 0.0 # Don't incetivize speed near obstacles
        else:
            # Reward = Normalized Speed * Alignment
            # If turning (alignment < 1), reward drops. If circling (alignment ~0), reward ~0.
            r_progress = 2.0 * (speed / self.target_speed) * alignment
            
            # Penalize speeding
            if speed > (self.target_speed + 5.0):
                 r_progress *= 0.5
        
        reward += r_progress
        
        # 2. Steering Stability Penalty
        # Penalize hard steering at high speeds to prevent oscillation
        steer = info.get('action_steer', 0.0)
        if speed > 1.0:
            r_stability = -1.0 * abs(steer) * (speed / 10.0)
            reward += r_stability
            
        # 3. Centering Penalty (Relaxed Linear)
        lat_err = abs(info.get('lateral_error', 0.0))
        r_centering = -1.0 * lat_err
        reward += r_centering
        
        # 4. Wrong Way Logic (Accumulator)
        if alignment < 0: # Facing backwards
            self.wrong_way_counter += 1
        else:
            self.wrong_way_counter = 0
            
        if self.wrong_way_counter > 50: # ~2-3 seconds wrong way
            reward = -100.0
            terminated = True
        
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
            
        # 5. Wrong Way / Shoulder - UNCHANGED (Lane Type Check)
        lane_type = info.get('lane_type', 'none')
        if lane_type != 'driving':
            reward -= 1.0
        
        # 6. Braking Reward - Encourage slowing down when obstacles near
        # Reward defensive driving: brake when scared, don't swerve
        min_dist = info.get('min_lidar_dist', 50.0)
        accel = info.get('action_accel', 0.0)
        if min_dist < 15.0 and accel < 0:  # Obstacle close AND braking
            # Small positive reward proportional to braking intensity
            r_brake = 0.2 * abs(accel)  # Max +0.2 for full brake
            reward += r_brake
            
        # 6. Terminal States
        if info.get('is_collision', False):
            reward = -100.0
            terminated = True
            
        if info.get('is_off_road', False):
             reward = -100.0
             terminated = True  # Stop immediately
        
        # Success (Survival Time)
        current_time = info.get('current_time', 0.0)
        if current_time > self.max_time:
             reward += 50.0
             truncated = True

        return reward, terminated, truncated
