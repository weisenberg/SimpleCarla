import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import os
import random
from scenario_map import OpenDriveParser
from traffic_manager import TrafficManager, EgoVehicle
from traffic_lights import IntersectionController

class SimpleCarlaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, map_name="Town01", render_mode=None, enable_traffic=False, traffic_density="high", enable_ego=False):
        self.map_name = map_name
        self.render_mode = render_mode
        self.enable_traffic = enable_traffic
        self.traffic_density = traffic_density
        self.enable_ego = enable_ego
        
        self.window = None
        self.clock = None
        # Double width for split screen if Ego enabled
        # reduced from 2048 to 1440 to fit screens
        self.screen_width = 1440 if enable_ego else 1024
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
        # We always create TrafficManager if Traffic OR Ego is enabled, to manage the Ego Vehicle properly
        self.traffic_manager = None
        if self.enable_traffic or self.enable_ego:
            self.traffic_manager = TrafficManager(self.parser)
            
            # Pass lights info
            all_lights = []
            for c in self.controllers:
                all_lights.extend(c.get_lights())
            self.traffic_manager.set_lights(all_lights)
            
            if self.enable_traffic:
                count = 70
                if self.traffic_density == "low": count = 15
                elif self.traffic_density == "mid": count = 35
                self.traffic_manager.spawn_vehicles(count)
                
            if self.enable_ego:
                self._spawn_ego()
        
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
        
        # Scale for Global View (Left side or Full)
        view_w = self.screen_width // 2 if self.enable_ego else self.screen_width
        
        scale_x = (view_w - 2*self.margin) / world_w if world_w > 0 else 1.0
        scale_y = (self.screen_height - 2*self.margin) / world_h if world_h > 0 else 1.0
        self.pixels_per_meter = min(scale_x, scale_y)

        # Placeholder Action/Obs (Modified observation space from snippet)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)

    def _spawn_ego(self):
        # Spawn EGO
        driving_lanes = [l for l in self.parser.lanes if l.type == 'driving']
        if driving_lanes:
            lane = random.choice(driving_lanes)
            length = len(lane.left_boundary)
            s = random.uniform(0, length - 5)
            # Create Ego
            config = {'color': (0, 255, 0), 'target_speed': 20.0, 'length': 4.5}
            self.ego_vehicle = EgoVehicle(9999, lane, s, 0, config)
            self.traffic_manager.vehicles.append(self.ego_vehicle)
        else:
            self.ego_vehicle = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset traffic if enabled (from snippet)
        if self.traffic_manager:
            self.traffic_manager.vehicles = []
            if self.enable_traffic:
                count = 70
                if self.traffic_density == "low": count = 15
                elif self.traffic_density == "mid": count = 35
                self.traffic_manager.spawn_vehicles(count)
            
            if self.enable_ego:
                self._spawn_ego()
                
        # Return observation matching new observation space
        return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8), {}

    def step(self, action):
        dt = 1.0/self.metadata["render_fps"]
        
        # Apply Action to Ego (if discrete/dict passed from run_env)
        # Assuming action is a dict or tuple: (throttle, steering)
        if self.enable_ego and self.ego_vehicle and isinstance(action, tuple):
             throttle, steering = action
             self.ego_vehicle.apply_control(throttle, steering)

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

        # --- View 1: Global Map (Left) ---
        view1_w = self.screen_width // 2 if self.enable_ego else self.screen_width
        
        # Coordinate Transform: Flip Y (Modified from snippet)
        def to_screen_global(x, y):
            # Scale to fit View1
            # We computed self.pixels_per_meter based on view1_w already?
            # Re-check init scale logic.
            # Scale was: view_w = 1024.
            # If we change width, we should update init or just use relative.
            # Let's use fixed offset for global view.
            
            sx = (x - self.min_x) * self.pixels_per_meter + self.margin
            sy = self.screen_height - ((y - self.min_y) * self.pixels_per_meter + self.margin)
            return int(sx), int(sy)
            
        self._draw_world(canvas, to_screen_global) # Use helper to draw world

        # --- View 2: Ego Centric (Right) ---
        if self.enable_ego and self.ego_vehicle:
            # Right Viewport
            view2_x = view1_w
            view2_w = self.screen_width - view1_w
            view2_center_x = view2_x + view2_w // 2
            view2_center_y = self.screen_height // 2
            
            # Clip/Clear Right Side
            right_rect = pygame.Rect(view2_x, 0, view2_w, self.screen_height)
            pygame.draw.rect(canvas, (20, 20, 20), right_rect) # Dark Grey Background for Ego View
            
            # Transform
            ego_x, ego_y, ego_h = self.ego_vehicle.get_position()
            # We want Ego at Center, Pointing North (Up).
            # Map Rotation: -ego_h
            # But wait, math angle 0 is East. Up is 90 (pi/2).
            # If Ego is heading h, we want to see it pointing Up. So we rotate world +90 deg.
            # Ego h=90 (North). We want to see it pointing Up. Rotate 0.
            # So Rotation Angle = 90 - deg(h).
            
            rotation_rad = math.pi/2 - ego_h
            
            ppm_ego = self.pixels_per_meter * 5.0 # Zoom in 5x for Ego View (Requested "Zoom In")
            
            def to_screen_ego(x, y):
                # 1. Translate relative to Ego
                dx = x - ego_x
                dy = y - ego_y
                
                # 2. Rotate
                rx = dx * math.cos(rotation_rad) - dy * math.sin(rotation_rad)
                ry = dx * math.sin(rotation_rad) + dy * math.cos(rotation_rad)
                
                # 3. Scale & Center (Flip Y for Screen)
                sx = view2_center_x + rx * ppm_ego
                sy = view2_center_y - ry * ppm_ego # Flip Y
                
                return int(sx), int(sy)
            
            # Set clip for drawing
            canvas.set_clip(right_rect)
            self._draw_world(canvas, to_screen_ego, ego_mode=True)
            canvas.set_clip(None) # Reset clip
            
            # Draw Separator
            pygame.draw.line(canvas, (255, 255, 255), (view1_w, 0), (view1_w, 768), 2)
            
            # --- Marker on Global Map ---
            # Ego is at ego_x, ego_y with heading ego_h
            mx, my = to_screen_global(ego_x, ego_y)
            # Draw a Triangle pointing in direction of heading
            # Heading is math angle (CCW from East)
            # Screen Y is flipped.
            # Triangle size
            size = 10
            
            # Tip: distance 'size' in direction ego_h
            # Left Wing: distance 'size' in direction ego_h + 140 deg
            # Right Wing: distance 'size' in direction ego_h - 140 deg
            
            # Screen Transform for direction vector (Y flip)
            # x' = x + cos(h) * size
            # y' = y - sin(h) * size
            
            tip = (mx + math.cos(ego_h) * size, my - math.sin(ego_h) * size)
            
            h_l = ego_h + math.radians(140)
            p_l = (mx + math.cos(h_l) * size, my - math.sin(h_l) * size)
            
            h_r = ego_h - math.radians(140)
            p_r = (mx + math.cos(h_r) * size, my - math.sin(h_r) * size)
            
            pygame.draw.polygon(canvas, (0, 255, 255), [tip, p_l, p_r]) # Cyan Triangle
            
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

    def _draw_world(self, canvas, to_screen_func, ego_mode=False):
        # Draw Lanes (Polygons) (Modified from snippet)
        for lane in self.parser.lanes:
            poly = lane.get_polygon()
            # Optimization: Check bounds? For now just draw all.
            if not poly: continue
            
            screen_poly = [to_screen_func(p[0], p[1]) for p in poly]
            
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
            
            line_width = 2
            pts = [to_screen_func(p[0], p[1]) for p in line.points]

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
                 
                 sx, sy = to_screen_func(lx, ly)
                 
                 color = (255, 0, 0)
                 if l.state == "GREEN": color = (0, 255, 0)
                 elif l.state == "YELLOW": color = (255, 255, 0)
                 
                 # Draw "Box" or Circle
                 radius = 3 if not ego_mode else 6 # Larger in Ego View
                 pygame.draw.circle(canvas, color, (sx, sy), radius)

        # Draw Vehicles (Added from snippet)
        if self.traffic_manager:
            for v in self.traffic_manager.vehicles:
                vx, vy, vh = v.get_position()
                sx, sy = to_screen_func(vx, vy)
                
                # Size
                # In Ego Mode: Scale size by PPM_EGO?
                # to_screen_func handles position. But size needs manual scaling used.
                # Re-calculate PPM from coordinate delta?
                # Or just pass PPM. 
                # Let's derive rough PPM from transforming (0,0) and (1,0) distance.
                p0 = to_screen_func(0,0)
                p1 = to_screen_func(1,0)
                current_ppm = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
                
                w = v.width * current_ppm
                l = v.length * current_ppm
                
                # Create rect
                surf = pygame.Surface((l, w), pygame.SRCALPHA)
                surf.fill(v.color)
                
                # Rotate
                # Heading h is in radians. Pygame rotate is degrees counter-clockwise.
                
                # GLOBAL VIEW:
                # Rot degrees = math.degrees(vh)
                
                # EGO VIEW:
                # The whole world is rotated by (Pi/2 - EgoH).
                # Vehicle heading relative to world is still vh.
                # So relative to screen: vh + (Pi/2 - EgoH).
                # Wait, canvas rotation adds.
                
                # Let's verify:
                # Ego Vehicle in Global: Heading North (Pi/2). Rot = 90. Point Up. Correct.
                # Ego Vehicle in Ego View:
                #   World Rot = Pi/2 - Pi/2 = 0.
                #   Ego V Heading = Pi/2. Screen Heading = 90. Points Up. Correct.
                
                # Other Vehicle East (0).
                #   World Rot = 0.
                #   Other V Heading = 0. Screen Heading = 0. Points Right. Correct.
                
                # So we just rotate surf by `vh + rotation_rad`?
                # Yes? Let's try.
                
                display_h = vh
                if ego_mode and self.ego_vehicle:
                     # Add map rotation
                     rotation_rad = math.pi/2 - self.ego_vehicle.get_position()[2]
                     display_h = vh + rotation_rad
                
                deg = math.degrees(display_h)
                rot_surf = pygame.transform.rotate(surf, deg)
                
                rect = rot_surf.get_rect(center=(sx, sy))
                canvas.blit(rot_surf, rect)
                
                # Highlight Ego
                if isinstance(v, EgoVehicle):
                     # Draw white outline rect
                     pygame.draw.rect(canvas, (255, 255, 255), rect, 2)


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
