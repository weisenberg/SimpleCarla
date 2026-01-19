import argparse
import time
import pygame
from simple_carla_env import SimpleCarlaEnv

def main():
    parser = argparse.ArgumentParser(description="Run SimpleCarla Environment")
    parser.add_argument("-map", type=str, default="Town01", help="Name of the town to load (e.g., Town01)")
    parser.add_argument('--traffic', nargs='?', const='high', default=None, help='Enable NPC traffic (low|mid|high). Default: high')
    args = parser.parse_args()
    
    enable_traffic = args.traffic is not None
    traffic_density = args.traffic

    env = SimpleCarlaEnv(map_name=args.map, render_mode="human", enable_traffic=enable_traffic, traffic_density=traffic_density)
    
    print(f"Running visualization for {args.map}. Press `Ctrl+C` to exit.")
    if enable_traffic:
        print(f"Traffic enabled: {traffic_density}")
    
    obs, info = env.reset()
    try:
        while True:
            # Handle Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            env.step(0) # Update simulation
            env.render()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
