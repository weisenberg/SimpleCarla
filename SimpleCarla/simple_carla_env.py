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
from rewards import RewardSignal

class SimpleCarlaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, map_name="Town01", render_mode=None, enable_traffic=False, traffic_density="high", enable_ego=False, enable_pedestrians=False, pedestrian_density="mid", sensors=None):
        self.map_name = map_name
        self.render_mode = render_mode
        self.enable_traffic = enable_traffic
        self.traffic_density = traffic_density
        self.enable_ego = enable_ego
        self.enable_pedestrians = enable_pedestrians
        self.pedestrian_density = pedestrian_density
        
        # Sensor Config
        self.sensors_config = sensors or {'lidar': True, 'collision': True, 'lane': True} # Default all on if not passed? Or off?
        # User wants to add by arg, so default off if not in dict, but if 'sensors' is passed, respect it.
        # run_env sets all to False by default if arg not present.
        
        self.reward_signal = RewardSignal()
        
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
        # We always create TrafficManager if Traffic OR Ego OR Pedestrians is enabled
        self.traffic_manager = None
        if self.enable_traffic or self.enable_ego or self.enable_pedestrians:
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
            
            if self.enable_pedestrians:
                p_count = 20
                if self.pedestrian_density == "low": p_count = 10
                elif self.pedestrian_density == "high": p_count = 50
                # Assuming TrafficManager has spawn_pedestrians
                if hasattr(self.traffic_manager, 'spawn_pedestrians'):
                    self.traffic_manager.spawn_pedestrians(p_count)
                
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

        # RL Config
        self.lidar_rays = 36
        self.lidar_range = 50.0

        # Visualization State
        self.latest_reward = 0.0
        self.total_reward = 0.0 # Track cumulative
        self.latest_info = {}
        self.latest_lidar = np.zeros((self.lidar_rays,), dtype=np.float32)

        # Placeholder Action/Obs (Modified observation space from snippet)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'minimap': spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8),
            'lidar': spaces.Box(low=0.0, high=self.lidar_range, shape=(self.lidar_rays,), dtype=np.float32),
            'speed': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'steering': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'lateral_error': spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),
            'heading_error': spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32)
        })

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
        self.reward_signal.reset()
        self.total_reward = 0.0
        
        # Reset traffic if enabled (from snippet)
        if self.traffic_manager:
            self.traffic_manager.vehicles = []
            if self.enable_traffic:
                count = 70
                if self.traffic_density == "low": count = 15
                elif self.traffic_density == "mid": count = 35
                self.traffic_manager.spawn_vehicles(count)
            
            if self.enable_pedestrians and hasattr(self.traffic_manager, 'spawn_pedestrians'):
                 if hasattr(self.traffic_manager, 'pedestrians'):
                     self.traffic_manager.pedestrians = []
                 
                 p_count = 20
                 if self.pedestrian_density == "low": p_count = 10
                 elif self.pedestrian_density == "high": p_count = 50
                 self.traffic_manager.spawn_pedestrians(p_count)

            if self.enable_ego:
                self._spawn_ego()
                
        # Return observation matching new observation space
        return self._get_obs(), {}

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
            
        # RL Physics / Metrics
        obs = self._get_obs()
        
        # Lane Metrics (Flag Check)
        lane_type = 'none'
        lat_err = 0.0
        head_err = 0.0
        
        if self.sensors_config.get('lane', False):
            if self.ego_vehicle:
                # Reuse from Obs if available, else recompute
                if 'lateral_error' in obs:
                     lat_err = float(obs['lateral_error'][0])
                     head_err = float(obs['heading_error'][0])
                lane_type, _, _ = self._get_lane_metrics()

        # Collision Check (Flag Check)
        is_collision = False 
        try:
             if self.enable_ego and self.ego_vehicle and self.sensors_config.get('collision', False):
                 # Vehicles
                 for v in self.traffic_manager.vehicles:
                     if v is not self.ego_vehicle:
                         vx, vy, _ = v.get_position()
                         dx = vx - self.ego_vehicle.x
                         dy = vy - self.ego_vehicle.y
                         # Vehicle Radius ~2m + Ego Radius ~2m = 4m threshold. Sq = 16.
                         if dx*dx + dy*dy < 16.0: 
                             is_collision = True
                             break
                 
                 # Pedestrians (Added)
                 if not is_collision and hasattr(self.traffic_manager, 'pedestrians'):
                     for p in self.traffic_manager.pedestrians:
                         dx = p.x - self.ego_vehicle.x
                         dy = p.y - self.ego_vehicle.y
                         # Ped Radius ~0.5m + Ego Radius ~2m = 2.5m threshold. Sq = 6.25.
                         if dx*dx + dy*dy < 6.25:
                             is_collision = True
                             break
        except Exception as e:
             # print(f"Collision Check Error: {e}")
             pass

        # Lead Distance Calculation
        dist_lead = 100.0
        if self.ego_vehicle:
            # Simple cone check or distance check in front
            # Vector: (cos(h), sin(h))
            ex, ey, eh = self.ego_vehicle.get_position()
            vx, vy = math.cos(eh), math.sin(eh)
            
            # Check Vehicles
            if self.traffic_manager:
                for v in self.traffic_manager.vehicles:
                    if v is not self.ego_vehicle:
                        tx, ty, _ = v.get_position()
                        dx, dy = tx - ex, ty - ey
                        
                        # Project onto heading vector
                        proj = dx * vx + dy * vy
                        
                        # Must be in front (proj > 0) and within cone (perp dist)
                        if proj > 0 and proj < dist_lead:
                            perp = abs(-vy * dx + vx * dy) # Cross product
                            if perp < 2.0: # Within lane width approx
                                dist_lead = proj
                
                # Check Pedestrians (radius 0.5)
                if hasattr(self.traffic_manager, 'pedestrians'):
                    for p in self.traffic_manager.pedestrians:
                        dx, dy = p.x - ex, p.y - ey
                        proj = dx * vx + dy * vy
                        if proj > 0 and proj < dist_lead:
                            perp = abs(-vy * dx + vx * dy)
                            if perp < 1.5: # Pedestrian can be narrower
                                dist_lead = proj - 0.5 # Surface distance
            
        is_off_road = (lane_type == 'none')
        is_red_light = self._check_red_light()
        
        info = {
            'speed': self.ego_vehicle.speed if self.ego_vehicle else 0.0,
            'is_collision': is_collision,
            'is_off_road': is_off_road,
            'lane_type': lane_type,
            'is_red_light': is_red_light,
            'distance_to_lead': dist_lead,
            'lateral_error': lat_err,
            'heading_error': head_err
        }
        
        # Reward
        reward, terminated, truncated = self.reward_signal.compute(info, dt)
        
        # Store for Visualization
        self.latest_reward = reward
        self.total_reward += reward
        self.latest_info = info
        self.latest_lidar = obs.get('lidar', np.zeros((self.lidar_rays,), dtype=np.float32))
        
        return obs, reward, terminated, truncated, info

    def _point_in_polygon(self, x, y, poly):
        # Ray casting algorithm
        if not poly: return False
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _get_lane_metrics(self):
        if not self.ego_vehicle: return 'none', 0.0, 0.0
        
        x, y, h = self.ego_vehicle.get_position()
        
        # 1. Find the Lane Polygon we are in
        best_lane = None
        
        for lane in self.parser.lanes:
            if not lane.left_boundary: continue
            
            # Check if point inside polygon
            poly = lane.get_polygon()
            if self._point_in_polygon(x, y, poly):
                best_lane = lane
                break
        
        if not best_lane:
             return 'none', 10.0, 0.0 # Off road
             
        # 2. Compute Metrics relative to Center Line
        # Lane Center Line = Average of Left and Right boundaries.
        # Find closest segment.
        
        min_dist = 1000.0
        lat_err = 0.0
        lane_heading = 0.0
        
        pts_l = best_lane.left_boundary
        pts_r = best_lane.right_boundary
        count = min(len(pts_l), len(pts_r))
        
        for i in range(count - 1):
            # Segment Center Points
            p0 = ((pts_l[i][0] + pts_r[i][0])/2, (pts_l[i][1] + pts_r[i][1])/2)
            p1 = ((pts_l[i+1][0] + pts_r[i+1][0])/2, (pts_l[i+1][1] + pts_r[i+1][1])/2)
            
            # Distance from Point (x,y) to Line Segment (p0, p1)
            # Project (x,y) onto line p0->p1
            px, py = p1[0]-p0[0], p1[1]-p0[1]
            norm = px*px + py*py
            if norm == 0: continue
            
            u = ((x - p0[0]) * px + (y - p0[1]) * py) / norm
            
            # Clamp u to segment [0, 1]
            u_clamped = max(min(u, 1), 0)
            
            # Closest point
            cx = p0[0] + u * px
            cy = p0[1] + u * py
            
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist:
                min_dist = dist
                
                # Determine sign of lateral error (Cross Product)
                # Global heading of road
                road_h = math.atan2(py, px)
                
                # Cross product (2D): A x B = AxBy - AyBx
                # Vector Road: (px, py)
                # Vector Center->Ego: (dx, dy) = (x-cx, y-cy)
                cross = px * dy - py * dx
                
                # If cross > 0, Ego is Left of road msg -> LatErr +
                lat_err = min_dist if cross > 0 else -min_dist
                
                # Heading Error
                # Check Driving Direction.
                # If best_lane.id > 0 (Left Lane), it drives AGAINST geometry indices.
                # So Road Heading is opposite of segment vector.
                
                # OpenDrive: Right lanes (ID < 0) move with geometry (0->N).
                # Left lanes (ID > 0) move against geometry (N->0).
                
                heading_ref = road_h
                if best_lane.id > 0: # Left Lane
                    heading_ref += math.pi
                    
                diff = h - heading_ref
                while diff > math.pi: diff -= 2*math.pi
                while diff < -math.pi: diff += 2*math.pi
                
                head_err = diff
                lane_heading = heading_ref

        return best_lane.type, lat_err, head_err

    def _check_red_light(self):
        # Determine if Ego is approaching a RED light
        if not self.ego_vehicle: return False
        if not self.controllers: return False
        
        x, y, h = self.ego_vehicle.get_position()
        
        # Optimize: Check distance to all TrafficLights
        for c in self.controllers:
            # Flatten lights from groups
            for group in c.groups:
                for light in group:
                    # light.pos is (x, y, heading)
                    lx, ly = light.pos[0], light.pos[1]
                    dist = math.sqrt((x-lx)**2 + (y-ly)**2)
                    
                    if dist < 15.0: # Approaching
                        if light.state == 'RED': # Uppercase "RED" from traffic_lights.py
                             return True
        return False
        
        if not best_lane:
             return 'none', 10.0, 0.0 # Off road
             
        # 2. Compute Metrics relative to Center Line
        # Lane Center Line = Average of Left and Right boundaries.
        # Find closest segment.
        
        min_dist = 1000.0
        lat_err = 0.0
        lane_heading = 0.0
        
        pts_l = best_lane.left_boundary
        pts_r = best_lane.right_boundary
        count = min(len(pts_l), len(pts_r))
        
        for i in range(count - 1):
            # Segment Center Points
            p0 = ((pts_l[i][0] + pts_r[i][0])/2, (pts_l[i][1] + pts_r[i][1])/2)
            p1 = ((pts_l[i+1][0] + pts_r[i+1][0])/2, (pts_l[i+1][1] + pts_r[i+1][1])/2)
            
            # Distance from Point (x,y) to Line Segment (p0, p1)
            # Project (x,y) onto line p0->p1
            px, py = p1[0]-p0[0], p1[1]-p0[1]
            norm = px*px + py*py
            if norm == 0: continue
            
            u = ((x - p0[0]) * px + (y - p0[1]) * py) / norm
            
            # Clamp u to segment [0, 1]
            u_clamped = max(min(u, 1), 0)
            
            # Closest point
            cx = p0[0] + u * px
            cy = p0[1] + u * py
            
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist:
                min_dist = dist
                
                # Determine sign of lateral error (Cross Product)
                # Global heading of road
                road_h = math.atan2(py, px)
                
                # Cross product of (Road Dir) x (Ego->Center) ?
                # Or simply:
                # Lat Err is positive if we are to the LEFT of the center line?
                # Transform Ego pos to Road Frame?
                
                # Cross product (2D): A x B = AxBy - AyBx
                # Vector Road: (px, py)
                # Vector Center->Ego: (dx, dy) = (x-cx, y-cy)
                cross = px * dy - py * dx
                
                # If cross > 0, Ego is Left of road msg -> LatErr +
                lat_err = min_dist if cross > 0 else -min_dist
                
                # Heading Error
                # Check Driving Direction.
                # If best_lane.id > 0 (Left Lane), it drives AGAINST geometry indices.
                # So Road Heading is opposite of segment vector.
                
                # OpenDrive: Right lanes (ID < 0) move with geometry (0->N).
                # Left lanes (ID > 0) move against geometry (N->0).
                
                heading_ref = road_h
                if best_lane.id > 0: # Left Lane
                    heading_ref += math.pi
                    # Also, Lateral Error definition might flip?
                    # Let's keep "Left of driving direction" as positive.
                    # If driving backwards, cross product needs check.
                    # Let's trust geometric left for now.
                    
                diff = h - heading_ref
                # Normalize to [-pi, pi]
                while diff > math.pi: diff -= 2*math.pi
                while diff < -math.pi: diff += 2*math.pi
                
                # Re-Check Lat Error Sign for Left Lane
                # If driving South, and we are East (Left) of center, lat err should be +?
                # Usually standard: Cross Track Error.
                # Let's stick to geometric cross product.
                
                head_err = diff
                lane_heading = heading_ref

        # Handle 'shoulder' type as bad? Yes, it's 'shoulder'.
        return best_lane.type, lat_err, head_err

    def _compute_lidar(self):
        if not self.ego_vehicle or not self.sensors_config.get('lidar', False):
            return np.zeros((self.lidar_rays,), dtype=np.float32)
            
        x, y, h = self.ego_vehicle.get_position()
        ranges = np.ones((self.lidar_rays,), dtype=np.float32) * self.lidar_range
        
        # Ray casting
        # Optimize: Check against Bounding Circles of Vehicles
        # Ego Obstacles: Traffic + Pedestrians
        obstacles = []
        if self.traffic_manager:
            for v in self.traffic_manager.vehicles:
                if v is not self.ego_vehicle:
                    vx, vy, _ = v.get_position()
                    obstacles.append({'x': vx, 'y': vy, 'r': 2.5}) 
            for p in self.traffic_manager.pedestrians:
                obstacles.append({'x': p.x, 'y': p.y, 'r': 0.5})
        
        if not obstacles:
             return ranges
             
        # Cast Rays
        # FOV: 360 degrees?
        for i in range(self.lidar_rays):
            angle = h + (i / self.lidar_rays) * 2 * math.pi
            rx = math.cos(angle)
            ry = math.sin(angle)
            
            min_d = self.lidar_range
            
            # Intersection with Circles
            # Ray: P = O + t*D
            # Circle: |P - C|^2 = R^2
            # |O + t*D - C|^2 = R^2
            # Let L = C - O
            # |t*D - L|^2 = R^2
            # t^2 |D|^2 - 2t(D.L) + |L|^2 - R^2 = 0
            # |D|=1. t^2 - 2t(D.L) + |L|^2 - R^2 = 0
            # Quadratic: at^2 + bt + c = 0
            # a = 1
            # b = -2(D.L)
            # c = |L|^2 - R^2
            
            for obs in obstacles:
                lx = obs['x'] - x
                ly = obs['y'] - y
                
                # Quick bounding box/distance check
                l_sq = lx*lx + ly*ly
                if l_sq > (self.lidar_range + obs['r'])**2:
                    continue
                    
                b = -2 * (rx * lx + ry * ly)
                c = l_sq - obs['r']**2
                
                delta = b*b - 4*c
                if delta >= 0:
                    sqrt_delta = math.sqrt(delta)
                    # Two solutions: t1, t2
                    t1 = (-b - sqrt_delta) / 2
                    t2 = (-b + sqrt_delta) / 2
                    
                    t = min(t1, t2)
                    if t < 0: t = max(t1, t2)
                    
                    if t > 0 and t < min_d:
                        min_d = t
            
            ranges[i] = min_d
            
        return ranges

    def _get_obs(self):
        img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        if self.render_mode == "rgb_array":
             img = self.render()
        
        # Lidar Logic
        lidar = self._compute_lidar()
        
        spd = 0.0
        steer = 0.0
        lat_err = 0.0
        head_err = 0.0
        
        if self.ego_vehicle:
            # Refresh pos from free roam state (ensure updated)
            ego_v = self.ego_vehicle
            spd = ego_v.speed
            steer = ego_v.steering
            
            if self.sensors_config.get('lane', False):
                lt, le, he = self._get_lane_metrics()
                lat_err = le
                head_err = he
            
        return {
            'minimap': img,
            'lidar': lidar,
            'speed': np.array([spd], dtype=np.float32),
            'steering': np.array([steer], dtype=np.float32),
            'lateral_error': np.array([lat_err], dtype=np.float32),
            'heading_error': np.array([head_err], dtype=np.float32)
        }

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
            
        self._draw_world(canvas, to_screen_global, self.pixels_per_meter) # Use helper to draw world

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
            self._draw_world(canvas, to_screen_ego, ppm_ego, ego_mode=True)
            self._draw_lidar(canvas, to_screen_ego, ego_mode=True) # Draw Lidar in Ego View
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

        # Draw HUD (Always on top)
        self._draw_hud(canvas)
            
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

    def _draw_world(self, canvas, to_screen_func, ppm, ego_mode=False):
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

        # Draw Pedestrians (NEW)
        if self.traffic_manager and hasattr(self.traffic_manager, 'pedestrians'):
             for p in self.traffic_manager.pedestrians:
                 px, py = p.x, p.y
                 sx, sy = to_screen_func(px, py)
                 # Color
                 col = (0, 255, 255) # Cyan
                 if p.mode == "CROSSING": col = (255, 0, 255) # Magenta for Jaywalking
                 
                 pygame.draw.circle(canvas, col, (sx, sy), 3 if not ego_mode else 5)

        # Draw Vehicles (Added from snippet)
        if self.traffic_manager:
            for v in self.traffic_manager.vehicles:
                vx, vy, vh = v.get_position()
                sx, sy = to_screen_func(vx, vy)
                
                # Size
                # Use passed PPM logic to avoid jerky integer snapping
                
                w = v.width * ppm
                l = v.length * ppm
                
                # Create rect
                surf = pygame.Surface((l, w), pygame.SRCALPHA)
                surf.fill(v.color)
                
                # Rotate
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

    def _draw_lidar(self, canvas, to_screen_func, ego_mode=False):
        if not self.ego_vehicle: return
        
        # Get Lidar Data
        ranges = self.latest_lidar
        if ranges is None or len(ranges) == 0:
             print("DEBUG: Lidar ranges empty or None")
             return
        
        # DEBUG CHECK
        # print(f"DEBUG: Drawing Lidar {len(ranges)} rays. Sample: {ranges[0]:.2f}")
        
        x, y, h = self.ego_vehicle.get_position()
        sx, sy = to_screen_func(x, y)
        
        # Draw Rays
        # N Rays distributed over 360 deg
        n = len(ranges)
        
        for i in range(n):
            dist = ranges[i]
            
            # Angle
            # Same logic as _compute_lidar
            angle = h + (i / n) * 2 * math.pi
            
            # End Point in World
            ex = x + dist * math.cos(angle)
            ey = y + dist * math.sin(angle)
            
            # Screen Coords
            esx, esy = to_screen_func(ex, ey)
            
            # Color
            # Green if max range (no hit), Red if hit
            color = (0, 255, 0)
            if dist < self.lidar_range - 0.1:
                color = (255, 0, 0)
                
            # Draw line
            pygame.draw.line(canvas, color, (sx, sy), (esx, esy), 1)
            
            # Draw dot at hit
            if dist < self.lidar_range - 0.1:
                 pygame.draw.circle(canvas, (255, 255, 0), (esx, esy), 2)

    def _draw_hud(self, canvas):
        if not pygame.font.get_init():
            pygame.font.init()
            
        font = pygame.font.SysFont("Arial", 18)
        
        info = self.latest_info
        if not info:
             # Draw Waiting text?
             # print("DEBUG: No INFO for HUD")
             pass
        
        # Lines to display
        # Use defaults if info missing
        lines = [
            f"Speed: {info.get('speed', 0):.1f} m/s" if info else "Speed: N/A",
            f"Crash: {info.get('is_collision', False)}" if info else "Crash: N/A",
            f"Reward: {self.total_reward:.2f}", # Show Cumulative
            f"Step Raw: {self.latest_reward:.2f}", # Show Raw Step
            f"Time Left: {self.reward_signal.max_episode_duration - self.reward_signal.current_episode_duration:.1f} s",
            f"Lane Type: {info.get('lane_type', 'none')}" if info else "Lane: N/A",
            f"Lat Err: {info.get('lateral_error', 0):.2f} m" if info else "Lat Err: N/A"
        ]
        
        # Draw on Right Side (Ego View Overlay)
        x = self.screen_width // 2 + 20
        y = 20
        
        # Background Box
        bg_rect = pygame.Rect(x-10, y-10, 200, len(lines)*25 + 20)
        s = pygame.Surface((bg_rect.w, bg_rect.h))
        s.set_alpha(180)
        s.fill((0, 0, 0))
        canvas.blit(s, bg_rect)
        
        try:
             for line in lines:
                 text = font.render(line, True, (255, 255, 255))
                 canvas.blit(text, (x, y))
                 y += 25
        except Exception as e:
             print(f"HUD Error: {e}")


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
