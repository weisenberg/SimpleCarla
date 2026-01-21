import argparse
import time
import pygame
from simple_carla_env import SimpleCarlaEnv

def main():
    parser = argparse.ArgumentParser(description="Run SimpleCarla Environment")
    parser.add_argument("-map", type=str, default="Town01", help="Name of the town to load (e.g., Town01)")
    parser.add_argument('--traffic', nargs='?', const='high', default=None, help='Enable NPC traffic (low|mid|high). Default: high')
    parser.add_argument('--pedestrians', nargs='?', const='mid', default=None, help='Enable Pedestrians (low|mid|high). Default: mid')
    parser.add_argument('--ego', action='store_true', help='Spawn Ego vehicle with WASD controls')
    parser.add_argument('--lidar', action='store_true', help='Enable Lidar Sensor')
    parser.add_argument('--collision', action='store_true', help='Enable Collision Detector')
    parser.add_argument('--lane', action='store_true', help='Enable Lane Sensor (Lateral Error)')
    parser.add_argument('--endless', action='store_true', help='Run in infinite episode mode (ignore collision termination)')
    args = parser.parse_args()
    
    enable_traffic = args.traffic is not None
    traffic_density = args.traffic
    
    enable_pedestrians = args.pedestrians is not None
    pedestrian_density = args.pedestrians
    
    enable_ego = args.ego
    
    sensors = {'lidar': args.lidar, 'collision': args.collision, 'lane': args.lane}

    print("Running visualization. Press Ctrl+C to exit.")

    env = SimpleCarlaEnv(map_name=args.map, render_mode="human", enable_traffic=enable_traffic, traffic_density=traffic_density, enable_pedestrians=enable_pedestrians, pedestrian_density=pedestrian_density, enable_ego=enable_ego, sensors=sensors, infinite_episode=args.endless)
    
    if enable_traffic:
        print("Traffic enabled:", traffic_density)
    if enable_pedestrians:
        print("Pedestrians enabled:", pedestrian_density)
    if enable_ego:
        print("Ego Agent enabled. Controls: W/S for Speed, A/D for Lane Change.")
    
    obs, info = env.reset()
    try:
        while True:
            throttle = 0.0
            lane_change = 0
            
            # Handle Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_a:
                        lane_change = -1
                    elif event.key == pygame.K_d:
                        lane_change = 1
            
            # Continuous Input
            keys = pygame.key.get_pressed()
            
            # Map to MultiDiscrete([3, 3])
            # Throttle: 0=Brake/Rev, 1=Idle, 2=Accel
            throttle_idx = 1 # Idle
            if keys[pygame.K_w]: throttle_idx = 2
            elif keys[pygame.K_s]: throttle_idx = 0
            
            # Steer: 0=Left, 1=Center, 2=Right
            # Note: 0 maps to -1.0 (Right Turn in our coord system?), 2 maps to +1.0 (Left Turn in our coord system?)
            # Logic: Positive Yaw is Left (CCW).
            # So we want A (Left) -> Positive Steering (+1.0) -> Index 2
            # We want D (Right) -> Negative Steering (-1.0) -> Index 0
            steer_idx = 1 # Center
            if keys[pygame.K_a]: steer_idx = 2 
            elif keys[pygame.K_d]: steer_idx = 0
            
            # Step with Action
            action = [throttle_idx, steer_idx] # Pass as list/array for MultiDiscrete
            obs, reward, terminated, truncated, info = env.step(action) 
            
            if terminated or truncated:
                reason = "Collision/Fail" if terminated else "Time Limit"
                print("Episode Finished. Reward: {:.2f} | Reason: {}".format(env.total_reward, reason))
                obs, info = env.reset()
            
            env.render()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
