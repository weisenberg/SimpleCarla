import argparse
import pygame
from simple_carla_env import SimpleCarlaEnv

def main():
    parser = argparse.ArgumentParser(description="Manual Control for SimpleCarla")
    parser.add_argument("-map", type=str, default="Town01", help="Map name")
    parser.add_argument("--traffic", nargs='?', const='high', default=None, help="Traffic density")
    parser.add_argument("--ego", action='store_true', help="Enable Ego Agent and Split Screen")
    
    args = parser.parse_args()
    
    # Initialize Env
    # Note: traffic argument validation logic
    enable_traffic = args.traffic is not None
    
    env = SimpleCarlaEnv(map_name=args.map, render_mode="human")
    
    # Custom attributes based on args
    env.enable_ego = args.ego
    if args.ego:
        env.screen_width = 1200
        env.screen_height = 600
    else:
        env.screen_width = 800
        env.screen_height = 800
        env.view_map = pygame.Surface((800, 800)) # Resize map view
    
    # Traffic setup
    if enable_traffic:
        count = 70
        if args.traffic == 'low': count = 15
        elif args.traffic == 'mid': count = 35
        env.traffic.spawn_vehicles(env.map, count)
    
    # Reset
    env.reset()
    if not args.ego:
        env.ego = None # Ensure no ego
    
    # Loop
    running = True
    steer = 0.0
    accel = 0.0
    
    print("Controls: W/S (Accel), A/D (Steer). ESC to exit.")
    
    while running:
        # Input
        keys = pygame.key.get_pressed()
        
        # Reset inputs
        accel = 0.0
        steer = 0.0
        
        if keys[pygame.K_w]: accel = 1.0
        if keys[pygame.K_s]: accel = -1.0
        if keys[pygame.K_a]: steer = -1.0
        if keys[pygame.K_d]: steer = 1.0
        
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Step
        env.step([steer, accel])
        
        # Render
        env.render()
        
        # HUD for Speed
        if env.ego and args.ego:
            speed = env.ego['v']
            text = env.font.render(f"Speed: {speed:.2f} m/s", True, (255, 255, 255))
            env.screen.blit(text, (10, 10))
            pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
