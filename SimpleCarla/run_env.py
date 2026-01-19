import argparse
import time
import pygame
from simple_carla_env import SimpleCarlaEnv

def main():
    parser = argparse.ArgumentParser(description="Run SimpleCarla Environment")
    parser.add_argument("-map", type=str, default="Town01", help="Name of the town to load (e.g., Town01)")
    parser.add_argument('--traffic', nargs='?', const='high', default=None, help='Enable NPC traffic (low|mid|high). Default: high')
    parser.add_argument('--ego', action='store_true', help='Spawn Ego vehicle with WASD controls')
    args = parser.parse_args()
    
    enable_traffic = args.traffic is not None
    traffic_density = args.traffic
    enable_ego = args.ego

    env = SimpleCarlaEnv(map_name=args.map, render_mode="human", 
                         enable_traffic=enable_traffic, traffic_density=traffic_density,
                         enable_ego=enable_ego)
    
    print(f"Running visualization for {args.map}. Press `Ctrl+C` to exit.")
    if enable_traffic:
        print(f"Traffic enabled: {traffic_density}")
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
            if keys[pygame.K_w]: throttle = 1.0
            if keys[pygame.K_s]: throttle = -1.0
            
            # Steering (Left is +Angle in standard math, but check Coordinate System)
            # Pygame Y is flipped.
            # EgoVehicle heading: Math Angle (CCW from East).
            # To turn Left (North): Increase Angle (+).
            # To turn Right (South): Decrease Angle (-).
            # Key A (Left) -> Positive Steer?
            # Key D (Right) -> Negative Steer?
            
            # Let's try:
            # A -> +1.0
            # D -> -1.0
            
            steering = 0.0
            if keys[pygame.K_a]: steering = 1.0 
            if keys[pygame.K_d]: steering = -1.0
            
            # Step with Action
            action = (throttle, steering)
            env.step(action) 
            env.render()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
