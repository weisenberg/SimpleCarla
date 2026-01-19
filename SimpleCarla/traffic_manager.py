import math
import random
import pygame

class Vehicle:
    def __init__(self, v_id, lane, s, speed, config):
        self.id = v_id
        self.lane = lane  # Lane object
        self.s = s        # Distance along the lane (meters)
        self.speed = speed # m/s (always positive magnitude)
        
        # OpenDrive: Right lanes (ID < 0) move with S (0 -> Length)
        # Left lanes (ID > 0) move against S (Length -> 0)
        self.direction = 1 if self.lane.id < 0 else -1
        
        self.length = config.get('length', 4.5)
        self.width = config.get('width', 2.0)
        self.color = config.get('color', (0, 0, 255))
        self.target_speed = config.get('target_speed', 13.0) # ~50km/h
        
        # IDM Parameters
        self.T = 1.5  # Safe time headway (Is)
        self.d0 = 2.0 # Minimum gap (m)
        self.a = 2.0  # Max acceleration (m/s^2)
        self.b = 2.0  # Comfortable deceleration (m/s^2)
        self.delta = 4 # Exponent
        
    def get_position(self):
        # Interpolate position
        pts = self.lane.left_boundary
        if not pts: return (0,0,0)
        
        # pts are ordered by s (0 -> Length)
        
        # Clamp s
        max_s = len(pts) - 1.0 # approx length
        draw_s = max(0.0, min(self.s, max_s))
        
        idx = int(draw_s)
        if idx >= len(pts) - 1: idx = len(pts) - 2
        
        # Get p0, p1 from left and right boundaries (average for center)
        # Note: Lane boundary lists are ordered by s.
        
        idx = max(0, idx)
        
        if idx >= len(self.lane.left_boundary) or idx >= len(self.lane.right_boundary):
             return (0,0,0)
        
        # Left/Right boundaries
        # OpenDrive: Left is +t, Right is -t relative to center line reference.
        # But our `Lane` object stores explicit boundary polygons.
        # `left_boundary` is the boundary on the 'left' side (w.r.t s direction).
        # `right_boundary` is on the 'right'.
        # Center of lane is average.
        
        pl0 = self.lane.left_boundary[idx]
        pr0 = self.lane.right_boundary[idx]
        p0 = ((pl0[0]+pr0[0])/2, (pl0[1]+pr0[1])/2)
        
        pl1 = self.lane.left_boundary[idx+1]
        pr1 = self.lane.right_boundary[idx+1]
        p1 = ((pl1[0]+pr1[0])/2, (pl1[1]+pr1[1])/2)
        
        residue = draw_s - idx
        
        x = p0[0] + (p1[0] - p0[0]) * residue
        y = p0[1] + (p1[1] - p0[1]) * residue
        
        # Heading of the ROAD geometry (s direction)
        heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        
        # If driving against S (Left lane), flip heading
        if self.direction == -1:
            heading += math.pi
            
        return x, y, heading

    def update(self, dt, lead_vehicle_gap):
        # IDM
        v = self.speed
        
        # Gap is already passed as relative distance (m) and lead_v_speed (m/s)
        s_gap = 1000.0
        dv = 0
        
        if lead_vehicle_gap:
            s_gap, lead_v_speed = lead_vehicle_gap
            dv = v - lead_v_speed
        
        s_star = self.d0 + v * self.T + (v * dv) / (2 * math.sqrt(self.a * self.b))
        
        # Avoid division by zero
        if s_gap < 0.1: s_gap = 0.1
        
        acc = self.a * (1 - (v / self.target_speed)**self.delta - (s_star / s_gap)**2)
        
        self.speed += acc * dt
        if self.speed < 0: self.speed = 0
        
        # Move
        self.s += self.speed * dt * self.direction

class EgoVehicle(Vehicle):
    def __init__(self, v_id, lane, s, speed, config):
        super().__init__(v_id, lane, s, speed, config)
        self.color = (0, 255, 0) # Green
        self.throttle = 0.0 
        self.steering = 0.0 # -1.0 (Left) to 1.0 (Right)
        
        # Free Roam State
        # Calculate initial x, y, h from lane/s
        self.x, self.y, self.h = super().get_position()
        self.free_roam_active = True

    def apply_control(self, throttle, steering):
        self.throttle = throttle
        self.steering = steering

    def get_position(self):
        # Override to return free roam state
        return self.x, self.y, self.h

    def update(self, dt, lead_vehicle_gap):
        # 1. Physics (Kinematic Bicycle Model)
        
        # Acceleration
        acc = self.throttle * 8.0 
        # Braking / Reverse Logic
        if (self.throttle > 0 and self.speed < 0) or (self.throttle < 0 and self.speed > 0):
             acc *= 2.0 
        
        self.speed += acc * dt
        
        # Cap Speed
        if self.speed > 40.0: self.speed = 40.0
        if self.speed < -15.0: self.speed = -15.0 
        
        # Steering (Yaw Rate)
        # Max steering angle roughly 45 deg? 
        # yaw_rate = (speed / wheelbase) * tan(steer_angle)
        # Let's simplify: Turn Rate proportional to steering & speed
        # But allow turning when slow too? No, cars need speed to turn.
        
        wheelbase = 3.0 # Approx
        max_steer_angle = math.radians(40)
        steer_angle = self.steering * max_steer_angle
        
        # Small speed threshold to avoid jitter at 0
        if abs(self.speed) > 0.1:
            # Kinematic Bicycle
            # beta = atan( l_r / l * tan(delta) ) ? Assuming Center of Mass.
            # Simplified: dH/dt = (v / L) * tan(delta)
            
            yaw_rate = (self.speed / wheelbase) * math.tan(steer_angle)
            self.h += yaw_rate * dt
        
        # Update Position
        self.x += self.speed * math.cos(self.h) * dt
        self.y += self.speed * math.sin(self.h) * dt
        
        # Note: We ignore 'lane' and 's' updates for Ego.
        # This means interactions with TrafficManager (lead vehicle) might break
        # if TrafficManager relies on 'lane' to find neighbors.
        # For now, we accept Ego is "Ghost" or we need to map-match.
        # Given "Free Roam", we likely accept Ghost for now.
        


class TrafficManager:
    def __init__(self, parser):
        self.parser = parser
        self.vehicles = []
        self.lanes_map = {} # (road_id, lane_id) -> Lane
        self._build_lane_map()
        
    def _build_lane_map(self):
        for lane in self.parser.lanes:
            if hasattr(lane, 'road_id'):
                 self.lanes_map[(lane.road_id, lane.id)] = lane
    
    def spawn_vehicles(self, n=50):
        driving_lanes = [l for l in self.parser.lanes if l.type == 'driving']
        if not driving_lanes: return
        
        for i in range(n):
            lane = random.choice(driving_lanes)
            length = len(lane.left_boundary)
            if length < 5: continue
            
            s = random.uniform(0, length - 5)
            # If left lane (id > 0), s travels L -> 0.
            # Random s is fine, direction handles movement.
            
            speed = random.uniform(0, 10.0)
            
            config = {
                'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)),
                'target_speed': random.uniform(8.0, 15.0), # 30-55 km/h
                'length': random.uniform(4.0, 5.0)
            }
            
            v = Vehicle(i, lane, s, speed, config)
            self.vehicles.append(v)

    def set_lights(self, lights):
        self.lights_map = {}
        for l in lights:
             self.lights_map[(l.road_id, l.lane_id)] = l

    def update(self, dt):
        for v in self.vehicles:
            # Skip Lane Change Logic for Ego (Free Roam)
            if isinstance(v, EgoVehicle):
                pass
            
            # Find lead vehicle
            min_dist = 1000.0
            lead_v_speed = 0
            
            # Simple linear search (optimize later with spatial hash if needed)
            for other in self.vehicles:
                if other.id == v.id: continue
                if other.lane == v.lane:
                    # Same lane check
                    # Calculate signed distance
                    # If dir=1 (0->L): dist = other.s - v.s
                    # If dir=-1 (L->0): dist = v.s - other.s
                    
                    dist = (other.s - v.s) * v.direction
                    
                    # We want positive dist (ahead)
                    if dist > 0:
                        # Bumper to bumper
                        gap = dist - (v.length/2 + other.length/2)
                        
                        if gap < min_dist:
                            min_dist = gap
                            lead_v_speed = other.speed
            
            # Check Traffic Light
            if hasattr(self, 'lights_map'):
                light = self.lights_map.get((v.lane.road_id, v.lane.id))
                if light and light.state != "GREEN":
                     # Stop at end of lane
                     lane_len = len(v.lane.left_boundary)
                     stop_s = lane_len - 1.0 if v.direction == 1 else 1.0
                     
                     raw_dist = (stop_s - v.s) * v.direction
                     
                     # Only stop if we haven't crossed the line yet
                     if raw_dist > 0:
                         dist_to_light = raw_dist - (v.length * 0.5 + 2.0) # buffer
                         
                         if dist_to_light < min_dist:
                             if dist_to_light < 0: dist_to_light = 0 # Stop immediately if within buffer
                             min_dist = dist_to_light
                             lead_v_speed = 0
            
            if min_dist < 0: min_dist = 0 # Crash?
            
            v.update(dt, (min_dist, lead_v_speed) if min_dist < 999 else None)
            
            # Transition Logic
            lane_len = len(v.lane.left_boundary) # approx
            
            next_road = None
            next_lane_id = None
            
            should_switch = False
            new_s = 0
            
            if v.direction == 1 and v.s >= lane_len - 1:
                # Right lane ended -> Successor
                if v.lane.road_successor:
                     next_road = v.lane.road_successor
                     next_lane_id = v.lane.raw_successor
                     should_switch = True
                     new_s = 0
                     
            elif v.direction == -1 and v.s <= 0 + 1:
                # Left lane ended -> Predecessor
                if v.lane.road_predecessor:
                     next_road = v.lane.road_predecessor
                     next_lane_id = v.lane.raw_predecessor
                     should_switch = True
                     new_s = -1
            
            if should_switch and next_road:
                # Calculate residue (distance traveled past the end of the lane)
                # Max S for current lane
                max_s_prev = len(v.lane.left_boundary) - 1.0
                residue = 0.0
                if v.direction == 1:
                    residue = max(0.0, v.s - max_s_prev)
                else:
                    residue = max(0.0, 0.0 - v.s)
                
                # Unpack with contact point support
                contact = 'start'
                if len(next_road) == 3:
                    rtype, rid, contact = next_road
                else:
                    rtype, rid = next_road
                
                if rtype == 'road':
                    # Fallback for missing ID (implicit link)
                    if next_lane_id is None:
                        next_lane_id = v.lane.id
                        
                    key = (rid, next_lane_id)
                    if key in self.lanes_map:
                        next_lane = self.lanes_map[key]
                        v.lane = next_lane
                        
                        # Update direction
                        v.direction = 1 if v.lane.id < 0 else -1
                        
                        # New Max S
                        max_s_new = len(next_lane.left_boundary) - 1.0
                        
                        # Set S based on Contact Point + Residue
                        if contact == 'start':
                             v.s = residue
                        elif contact == 'end':
                             v.s = max_s_new - residue
                        else:
                             # Fallback
                             v.s = residue if v.direction == 1 else max_s_new - residue

                    else:
                        v.speed = 0 # Dead end
                        print(f"Vehicle {v.id} stopped: Dead End at Road {v.lane.road_id} Lane {v.lane.id}. Next Key {key} not found.")
                else:
                    # Junction handling via parser routes
                    found_route = False
                    curr_road_id = v.lane.road_id
                    
                    if hasattr(self.parser, 'junction_routes') and curr_road_id in self.parser.junction_routes:
                        possible_routes = self.parser.junction_routes[curr_road_id]
                        # Filter routes that have a link for our lane
                        valid_routes = []
                        for conn_road, links in possible_routes:
                             if v.lane.id in links:
                                 valid_routes.append((conn_road, links[v.lane.id]))
                        
                        if valid_routes:
                            target_road_id, target_lane_id = random.choice(valid_routes)
                            next_key = (target_road_id, target_lane_id)
                            
                            if next_key in self.lanes_map:
                                v.lane = self.lanes_map[next_key]
                                v.direction = 1 if v.lane.id < 0 else -1
                                
                                # Junction roads usually flow 0->S.
                                if v.direction == 1:
                                     v.s = 0.5
                                else:
                                     v.s = len(v.lane.left_boundary) - 1.5
                                
                                found_route = True
                    
                    if not found_route:
                        # Dead end or no path found
                        v.speed = 0
                        v.s = lane_len - 1 if v.direction == 1 else 0
                        print(f"Vehicle {v.id} stopped: Failed Junction Transition from {curr_road_id} Lane {v.lane.id}")
            
            # Simple boundary check to prevent out of bounds
            if v.s < 0: v.s = 0; v.speed = 0
            if v.s > lane_len: 
                 v.s = lane_len; v.speed = 0
                 # print(f"Vehicle {v.id} stopped: Boundary {v.lane.road_id}")
