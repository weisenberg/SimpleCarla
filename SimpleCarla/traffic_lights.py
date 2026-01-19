import random

class TrafficLight:
    def __init__(self, road_id, lane_id, pos, state="RED"):
        self.road_id = road_id
        self.lane_id = lane_id
        self.pos = pos # (x, y, heading)
        self.state = state # "RED", "YELLOW", "GREEN"

class IntersectionController:
    def __init__(self, junction_id, incoming_roads, parser):
        self.junction_id = junction_id
        self.incoming_roads = incoming_roads # list of road IDs
        self.parser = parser
        
        # Group lights by Incoming Road
        # self.groups = [ [Light1, Light2], [Light3, Light4] ... ]
        self.groups = []
        self._build_lights()
        
        self.current_group_idx = 0
        self.timer = 0
        self.phase = "GREEN" # GREEN -> YELLOW -> RED (switch)
        
        # Config
        self.green_duration = 3.0 # Reduced from 5.0 for faster cycling
        self.yellow_duration = 2.0
        self.all_red_duration = 1.0 # Reduced from 2.0
        
        # Initial State
        self._apply_state()

    def _build_lights(self):
        # identifying lanes that ENTER the junction
        # For each incoming road, find driving lanes that flow INTO the junction
        # We need to know the 'contact point'.
        # Heuristic: 
        # If Road Successor is this Junction -> Right Lanes (ID < 0) enter.
        # If Road Predecessor is this Junction -> Left Lanes (ID > 0) enter.
        
        # This requires checking the ROAD properties in Parser.
        # Parser stores lanes flat.
        # We can scan parser.lanes.
        
        for rid in self.incoming_roads:
            # Find all lanes for this road
            lanes = [l for l in self.parser.lanes if hasattr(l, 'road_id') and l.road_id == rid and l.type == 'driving']
            
            group_lights = []
            
            # Check parsing data (we need to access road connectivity from parser?)
            # The parser doesn't store Road objects in a dict suitable for easy connectivity lookup of roads.
            # But Lane objects have `road_successor` and `road_predecessor` tuples (type, id).
            
            # We can inspect one lane from the road to see connection.
            if not lanes: continue
            
            # Let's check which end connects to THIS junction
            # Road's Successor == (junction, self.junction_id)?
            # Road's Predecessor == (junction, self.junction_id)?
            
            # Since all lanes in a road share connectivity, check first one
            l_rep = lanes[0] 
            
            enters_succ = False
            enters_pred = False
            
            if l_rep.road_successor and l_rep.road_successor[0] == 'junction' and l_rep.road_successor[1] == self.junction_id:
                enters_succ = True
            if l_rep.road_predecessor and l_rep.road_predecessor[0] == 'junction' and l_rep.road_predecessor[1] == self.junction_id:
                enters_pred = True
                
            # Filter lanes that actually drive TOWARDS that end
            relevant_lanes = []
            for l in lanes:
                # Right lane (ID < 0): Moves 0->L. Enters Successor.
                if l.id < 0 and enters_succ:
                    relevant_lanes.append(l)
                # Left lane (ID > 0): Moves L->0. Enters Predecessor.
                if l.id > 0 and enters_pred:
                    relevant_lanes.append(l)
            
            if not relevant_lanes: continue
            
            for l in relevant_lanes:
                # Position of light is at the END of flow.
                # If ID < 0: End of lane (last point of boundary).
                # If ID > 0: Start of lane (first point of boundary, effectively index 0).
                
                # Use Right Boundary (Inner)? or Left (Outer)?
                # Usually lights are on the Left or Right? 
                # Let's put it on the Left Boundary (outer side) or Center?
                # Center is safer.
                
                pt = (0,0)
                if l.id < 0:
                    # End
                    pts_l = l.left_boundary
                    pts_r = l.right_boundary
                    if not pts_l: continue
                    pl = pts_l[-1]
                    pr = pts_r[-1]
                    pt = ((pl[0]+pr[0])/2, (pl[1]+pr[1])/2)
                else:
                    # Start (since flow is Reversed)
                    # wait, left_boundary is geometric sequence.
                    # ID > 0 moves against geometry.
                    # So it enters junction at s=0.
                    pts_l = l.left_boundary
                    pts_r = l.right_boundary
                    if not pts_l: continue
                    pl = pts_l[0]
                    pr = pts_r[0]
                    pt = ((pl[0]+pr[0])/2, (pl[1]+pr[1])/2)
                
                group_lights.append(TrafficLight(rid, l.id, pt))
                
            if group_lights:
                self.groups.append(group_lights)

    def update(self, dt):
        self.timer += dt
        
        if self.phase == "GREEN":
            if self.timer >= self.green_duration:
                self.phase = "YELLOW"
                self.timer = 0
                self._apply_state()
                
        elif self.phase == "YELLOW":
            if self.timer >= self.yellow_duration:
                self.phase = "ALL_RED"
                self.timer = 0
                self._apply_state()
                
        elif self.phase == "ALL_RED":
            if self.timer >= self.all_red_duration:
                # Switch to next group
                self.current_group_idx = (self.current_group_idx + 1) % len(self.groups)
                self.phase = "GREEN"
                self.timer = 0
                self._apply_state()

    def _apply_state(self):
        # Set all to Red first
        for g in self.groups:
            for l in g:
                l.state = "RED"
                
        # Set current group to Phase IF phase is GREEN or YELLOW
        if self.phase in ["GREEN", "YELLOW"] and self.groups:
            current_g = self.groups[self.current_group_idx]
            for l in current_g:
                l.state = self.phase
        
            for l in current_g:
                l.state = self.phase
                
    def get_lights(self):
        all_lights = []
        for g in self.groups:
            all_lights.extend(g)
        return all_lights
