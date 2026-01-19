import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import os
from scenario_map import OpenDriveParser
from traffic_manager import TrafficManager
from traffic_lights import IntersectionController

class SimpleCarlaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, map_name="Town01", render_mode=None, enable_traffic=False, traffic_density="high"):
        self.map_name = map_name
        self.render_mode = render_mode
        self.enable_traffic = enable_traffic
        self.traffic_density = traffic_density
        self.window = None
        self.clock = None
        self.screen_width = 1024
        self.screen_height = 768
        self.margin = 50
        
        pygame.init()
        pygame.display.set_caption("SimpleCarla - " + map_name)
        
        # Load Map
        # Assuming map is in standard location relative to this file or CWD
        # The user runs from workspace root, maps are in Maps/CARLA
        # valid paths: Maps/CARLA/{map_name}.xodr
        self.map_path = f"Maps/CARLA/{map_name}.xodr"
        if not os.path.exists(self.map_path):
             # Try absolute path fallback (user workspace)
             self.map_path = f"/Users/ali/Desktop/uni/master thesis/playground/Carla/Maps/CARLA/{map_name}.xodr"
        
        print(f"Loading map: {self.map_path}")
        self.parser = OpenDriveParser(self.map_path) # Changed from map_file to map_path
        self.parser.parse()
        self.lanes = self.parser.lanes
        
        # Traffic Lights
        self.controllers = []
        if hasattr(self.parser, 'junctions'):
            for jid, roads in self.parser.junctions.items():
                self.controllers.append(IntersectionController(jid, roads, self.parser))
        
        # Traffic Manager
        self.traffic_manager = None
        if self.enable_traffic:
            self.traffic_manager = TrafficManager(self.parser)
            
            # Pass lights info
            all_lights = []
            for c in self.controllers:
                all_lights.extend(c.get_lights())
            self.traffic_manager.set_lights(all_lights)
            
            count = 70
            if self.traffic_density == "low": count = 15
            elif self.traffic_density == "mid": count = 35
            
            self.traffic_manager.spawn_vehicles(count)
        
        # Calculate bounds for rendering (Modified logic from snippet)
        # Using lines for bounds as in snippet, but ensuring points exist
        all_x = []
        all_y = []
        for l in self.parser.lines:
            if l.points:
                all_x.extend([p[0] for p in l.points])
                all_y.extend([p[1] for p in l.points])
        
        if all_x and all_y:
            self.min_x = min(all_x)
            self.max_x = max(all_x)
            self.min_y = min(all_y)
            self.max_y = max(all_y)
        else: # Fallback if no lines found
            self.min_x, self.min_y = -100, -100
            self.max_x, self.max_y = 100, 100
            
        # Compute Scale (Modified logic from snippet)
        world_w = self.max_x - self.min_x
        world_h = self.max_y - self.min_y
        scale_x = (self.screen_width - 2*self.margin) / world_w if world_w > 0 else 1.0
        scale_y = (self.screen_height - 2*self.margin) / world_h if world_h > 0 else 1.0
        self.pixels_per_meter = min(scale_x, scale_y)

        # Placeholder Action/Obs (Modified observation space from snippet)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset traffic if enabled (from snippet)
        if self.enable_traffic and self.traffic_manager:
            self.traffic_manager.vehicles = []
            count = 70
            if self.traffic_density == "low": count = 15
            elif self.traffic_density == "mid": count = 35
            self.traffic_manager.spawn_vehicles(count)
        # Return observation matching new observation space
        return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8), {}

    def step(self, action):
        dt = 1.0/self.metadata["render_fps"]
        
        # Update Traffic (Added from snippet)
        if self.traffic_manager:
            self.traffic_manager.update(dt) 

        # Update Lights
        for c in self.controllers:
            c.update(dt)
            
        # Return observation matching new observation space
        return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8), 0.0, False, False, {}

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height)) # Changed to screen_width/height
            pygame.display.set_caption(f"SimpleCarla - {self.map_name}") # Kept map_name
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.screen_width, self.screen_height)) # Changed to screen_width/height
        canvas.fill((0, 100, 0)) # Green background

        # Coordinate Transform: Flip Y (Modified from snippet)
        def to_screen(x, y):
            sx = (x - self.min_x) * self.pixels_per_meter + self.margin
            sy = self.screen_height - ((y - self.min_y) * self.pixels_per_meter + self.margin)
            return int(sx), int(sy)

        # Draw Lanes (Polygons) (Modified from snippet)
        for lane in self.parser.lanes:
            poly = lane.get_polygon()
            if not poly: continue
            
            screen_poly = [to_screen(p[0], p[1]) for p in poly]
            
            color = (128, 128, 128) # Default Asphalt
            if lane.type == 'shoulder':
                color = (100, 100, 100) # Darker
            elif lane.type == 'sidewalk':
                color = (200, 200, 200) # Light Grey
            elif lane.type == 'driving':
                color = (50, 50, 50) # Dark Asphalt
            elif lane.type == 'none':
                if lane.is_junction:
                     color = (60, 60, 60) # Slightly lighter for junction
                else:
                     continue
            
            pygame.draw.polygon(canvas, color, screen_poly)
            
        # Draw Lines
        for line in self.parser.lines:
            if len(line.points) < 2: continue
            
            # Color
            c = (255, 255, 255) # White
            if line.color == "yellow":
                c = (255, 255, 0)
            elif line.color == "blue":
                c = (0, 0, 255)
            elif line.color == "green":
                c = (0, 255, 0)
            elif line.color == "red":
                c = (255, 0, 0)
            
            # Handle Junction Guides (Modified from snippet)
            is_guide = False
            if line.type == "none" and line.is_junction:
                is_guide = True
                c = (180, 180, 180) # Grey
            
            should_draw_solid = (line.type == "solid")
            should_draw_broken = (line.type in ["broken", "dashed", "dotted"]) or is_guide
            
            line_width = 1
            pts = [to_screen(p[0], p[1]) for p in line.points]

            if should_draw_solid and not is_guide:
                pygame.draw.lines(canvas, c, False, pts, line_width)
            elif should_draw_broken:
                # Dense points check (from snippet)
                 for i in range(0, len(pts)-1, 2):
                    pygame.draw.line(canvas, c, pts[i], pts[i+1], line_width)
            else:
                 # Solid default (from snippet)
                 pygame.draw.lines(canvas, c, False, pts, line_width)
                 
                 
        # Draw Traffic Lights
        for c in self.controllers:
             for l in c.get_lights():
                 lx, ly = l.pos
                 if lx == 0 and ly == 0: continue # Invalid pos
                 
                 sx, sy = to_screen(lx, ly)
                 
                 color = (255, 0, 0)
                 if l.state == "GREEN": color = (0, 255, 0)
                 elif l.state == "YELLOW": color = (255, 255, 0)
                 
                 # Draw "Box" or Circle
                 radius = 3
                 pygame.draw.circle(canvas, color, (sx, sy), radius)

        # Draw Vehicles (Added from snippet)
        if self.traffic_manager:
            for v in self.traffic_manager.vehicles:
                vx, vy, vh = v.get_position()
                sx, sy = to_screen(vx, vy)
                
                # Size
                w = v.width * self.pixels_per_meter
                l = v.length * self.pixels_per_meter
                
                # Create rect
                surf = pygame.Surface((l, w), pygame.SRCALPHA)
                surf.fill(v.color)
                
                # Rotate
                # Heading h is in radians. Pygame rotate is degrees counter-clockwise.
                # If h is math angle (CCW from X), then angle = h * 180/pi.
                # But screen coordinate Y is flipped.
                # Rotation must account for Y-flip. A math angle of +45 (Up-Right) becomes Down-Right on screen if we flip Y?
                # Actually, standard math rotation matches if we rotate visual element properly.
                # Let's try simple degrees conversion first.
                
                deg = math.degrees(vh)
                rot_surf = pygame.transform.rotate(surf, deg)
                
                rect = rot_surf.get_rect(center=(sx, sy))
                canvas.blit(rot_surf, rect)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update() # Added update
            self.clock.tick(self.metadata["render_fps"])
            return None # Changed return for human mode
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
