import xml.etree.ElementTree as ET
import numpy as np
import math

class Lane:
    def __init__(self, lane_id, lane_type, left_boundary, right_boundary, is_junction=False, road_id=None, predecessor_id=None, successor_id=None):
        self.id = lane_id
        self.type = lane_type
        self.left_boundary = left_boundary   # list of (x, y)
        self.right_boundary = right_boundary # list of (x, y)
        self.is_junction = is_junction
        self.road_id = road_id
        self.predecessor_id = predecessor_id # (road_id, lane_id) or just lane_id? 
        # Actually, in OpenDrive, lane link just gives the ID. 
        # The Road link determines the Road ID.
        # But for the Agent, we want `(next_road_id, next_lane_id)`.
        # So let's store `raw_predecessor` (id) and `raw_successor` (id) 
        # And later we can resolve them.
        # But wait, `predecessor_id` in Lane Link is the ID of the lane in the PREDECESSOR ROAD.
        
        self.raw_predecessor = predecessor_id 
        self.raw_successor = successor_id
        
        # We will compute these computed global links:
        self.successors = [] # list of (road_id, lane_id)
        self.predecessors = [] # list of (road_id, lane_id)


    def get_polygon(self):
        # Create a closed polygon from boundaries
        if not self.left_boundary or not self.right_boundary:
            return []
        # Right boundary needs to be reversed to form a loop
        return self.left_boundary + self.right_boundary[::-1]

class RoadLine:
    def __init__(self, points, line_type, color, is_junction=False):
        self.points = points
        self.type = line_type
        self.color = color
        self.is_junction = is_junction

class OpenDriveParser:
    def __init__(self, map_file):
        self.tree = ET.parse(map_file)
        self.root = self.tree.getroot()
        self.lanes = []  # list of Lane objects
        self.lines = []  # list of RoadLine objects

    def get_width(self, width_objs, ds):
        target_rec = None
        for w in width_objs:
            if w['sOffset'] <= ds:
                target_rec = w
            else:
                break
        
        if target_rec is None:
            return 0.0
            
        ds_local = ds - target_rec['sOffset']
        return target_rec['a'] + target_rec['b']*ds_local + target_rec['c']*ds_local**2 + target_rec['d']*ds_local**3

    def parse(self):
        new_lanes = []
        new_lines = []
        
        # Parse Junctions
        self.junctions = {} # id -> list of incoming road ids
        self.junction_routes = {} # incoming_road_id -> list of (connecting_road_id, lane_links_dict)
        
        for junc in self.root.findall('junction'):
            j_id = int(junc.get('id'))
            incoming_roads = set()
            for conn in junc.findall('connection'):
                inc_road = int(conn.get('incomingRoad'))
                incoming_roads.add(inc_road)
                
                conn_road = int(conn.get('connectingRoad'))
                links = {}
                for ll in conn.findall('laneLink'):
                    links[int(ll.get('from'))] = int(ll.get('to'))
                
                if inc_road not in self.junction_routes:
                    self.junction_routes[inc_road] = []
                self.junction_routes[inc_road].append((conn_road, links))
                
            self.junctions[j_id] = list(incoming_roads)

        for road in self.root.findall('road'):

            plan_view = road.find('planView')
            geometries = plan_view.findall('geometry')
            geometries.sort(key=lambda g: float(g.get('s')))
            
            geo_ranges = []
            for g in geometries:
                s = float(g.get('s'))
                l = float(g.get('length'))
                geo_ranges.append({'s': s, 'end': s + l, 'elem': g})
                
            lanes = road.find('lanes')
            if lanes is None: continue
            
            # Parse lane offsets
            lane_offsets = []
            if lanes is not None:
                for lo in lanes.findall('laneOffset'):
                    lane_offsets.append({
                        's': float(lo.get('s')),
                        'a': float(lo.get('a')),
                        'b': float(lo.get('b')),
                        'c': float(lo.get('c')),
                        'd': float(lo.get('d'))
                    })
            lane_offsets.sort(key=lambda x: x['s'])

            road_id = int(road.get('id'))
            road_predecessor = None
            road_successor = None
            
            link = road.find('link')
            if link is not None:
                pred = link.find('predecessor')
                if pred is not None:
                    road_predecessor = (pred.get('elementType'), int(pred.get('elementId')), pred.get('contactPoint'))
                succ = link.find('successor')
                if succ is not None:
                    road_successor = (succ.get('elementType'), int(succ.get('elementId')), succ.get('contactPoint'))
            
            lane_sections = lanes.findall('laneSection')
            lane_sections.sort(key=lambda ls: float(ls.get('s')))
            road_length = float(road.get('length'))
            junction_val = road.get('junction')
            is_junction = (junction_val is not None and junction_val != "-1")
            
            for i, ls in enumerate(lane_sections):
                s_section = float(ls.get('s'))
                if i < len(lane_sections) - 1:
                    s_section_end = float(lane_sections[i+1].get('s'))
                else:
                    s_section_end = road_length
                
                # Center lane mark (Lane 0)
                center_lane_mark_type = "none"
                center_lane_color = "white"
                center_node = ls.find('center')
                if center_node is not None:
                    l0 = center_node.find('lane')
                    if l0 is not None:
                        rm0 = l0.find('roadMark')
                        if rm0 is not None:
                            center_lane_mark_type = rm0.get('type') or "none"
                            center_lane_color = rm0.get('color') or "white"

                def process_side(side_node, side_sign, add_center_line): 
                    if side_node is None: return
                    
                    ls_lanes = side_node.findall('lane')
                    # Sort by abs(id) ascending (closest to center first)
                    ls_lanes.sort(key=lambda l: abs(int(l.get('id'))))
                    
                    # Store width funcs & markings
                    current_width_funcs = []
                    lane_types = []
                    lane_ids = []
                    lane_marks = [] # (type, color)
                    lane_links = [] # (pred, succ) IDs
                    
                    for lane in ls_lanes:
                        l_id = int(lane.get('id'))
                        l_type = lane.get('type')
                        w_elements = lane.findall('width')
                        w_objs = []
                        for w in w_elements:
                            w_objs.append({
                                'sOffset': float(w.get('sOffset')),
                                'a': float(w.get('a')),
                                'b': float(w.get('b')),
                                'c': float(w.get('c')),
                                'd': float(w.get('d'))
                            })
                        w_objs.sort(key=lambda x: x['sOffset'])
                        current_width_funcs.append(w_objs)
                        lane_types.append(l_type)
                        lane_ids.append(l_id)
                        
                        # Markings (Outer edge)
                        rm = lane.find('roadMark')
                        if rm is not None:
                           lane_marks.append((rm.get('type') or "none", rm.get('color') or "white"))
                        else:
                           lane_marks.append(("none", "white"))
                           
                        # Links
                        l_link = lane.find('link')
                        pred_id = None
                        succ_id = None
                        if l_link is not None:
                            lp = l_link.find('predecessor')
                            if lp is not None: pred_id = int(lp.get('id'))
                            ls_succ = l_link.find('successor')
                            if ls_succ is not None: succ_id = int(ls_succ.get('id'))
                        lane_links.append((pred_id, succ_id))

                    step = 1.0 # meters
                    
                    # Generate s_samples ensuring the end point is included
                    s_samples = []
                    curr = s_section
                    while curr < s_section_end - 1e-6:
                        s_samples.append(curr)
                        curr += step
                    s_samples.append(s_section_end)
                    
                    # We need N+1 boundaries for N lanes
                    boundaries = [ [] for _ in range(len(ls_lanes) + 1) ]
                    
                    for s_curr in s_samples:
                        
                        # Find geometry
                        geo = None
                        for gr in geo_ranges:
                            if gr['s'] <= s_curr <= gr['end'] + 0.1: 
                                geo = gr
                                break
                        
                        if geo is None and geo_ranges:
                            # Extrapolate to avoid gaps that cause visual jumping
                            valid_geos = [gr for gr in geo_ranges if gr['s'] <= s_curr + 1.0]
                            if valid_geos:
                                geo = valid_geos[-1]
                            else:
                                geo = geo_ranges[0]
                        
                        if geo is None: continue 
                        
                        ds_geo = s_curr - geo['s']
                        g_elem = geo['elem']
                        
                        x0 = float(g_elem.get('x'))
                        y0 = float(g_elem.get('y'))
                        h0 = float(g_elem.get('hdg'))
                        
                        rx, ry, rh = x0, y0, h0

                        # Geometry calculation
                        if g_elem.find('line') is not None:
                            rx = x0 + ds_geo * math.cos(h0)
                            ry = y0 + ds_geo * math.sin(h0)
                            rh = h0
                        elif g_elem.find('arc') is not None:
                            c = float(g_elem.find('arc').get('curvature'))
                            rh = h0 + c * ds_geo
                            if abs(c) > 1e-10:
                                rx = x0 + (math.sin(rh) - math.sin(h0))/c
                                ry = y0 + (math.cos(h0) - math.cos(rh))/c
                            else:
                                rx = x0 + ds_geo * math.cos(h0)
                                ry = y0 + ds_geo * math.sin(h0)
                        elif g_elem.find('spiral') is not None:
                             curv_start = float(g_elem.find('spiral').get('curvStart'))
                             curv_end = float(g_elem.find('spiral').get('curvEnd'))
                             length = float(g_elem.get('length'))
                             c_dot = (curv_end - curv_start) / length
                             n_sub = 5
                             d_sub = ds_geo / n_sub
                             temp_x, temp_y, temp_h = x0, y0, h0
                             cur_s = 0
                             for _ in range(n_sub):
                                 k = curv_start + c_dot * (cur_s + d_sub/2)
                                 temp_h += k * d_sub
                                 temp_x += d_sub * math.cos(temp_h)
                                 temp_y += d_sub * math.sin(temp_h)
                                 cur_s += d_sub
                             rx, ry, rh = temp_x, temp_y, temp_h
                        else:
                            rx = x0 + ds_geo * math.cos(h0)
                            ry = y0 + ds_geo * math.sin(h0)
                            rh = h0

                        nx = -math.sin(rh)
                        ny = math.cos(rh)
                        
                        # Apply Lane Offset
                        current_offset = 0.0
                        for lo in lane_offsets:
                            if lo['s'] <= s_curr:
                                ds_offset = s_curr - lo['s']
                                current_offset = lo['a'] + lo['b']*ds_offset + lo['c']*ds_offset**2 + lo['d']*ds_offset**3
                            else:
                                break
                        
                        rx += nx * current_offset
                        ry += ny * current_offset
                        
                        accum_width = 0.0
                        
                        # Boundary 0 (Center)
                        bx = rx
                        by = ry
                        boundaries[0].append((bx, by))
                        
                        for k, w_funcs in enumerate(current_width_funcs):
                            w = self.get_width(w_funcs, s_curr - s_section)
                            accum_width += w
                            
                            eff_width = accum_width if side_sign > 0 else -accum_width
                            
                            bx = rx + nx * eff_width
                            by = ry + ny * eff_width
                            
                            boundaries[k+1].append((bx, by))
                    
                    # Create Lanes
                    for k in range(len(ls_lanes)):
                        # Boundary K is inner (right side of lane), Boundary K+1 is outer (left side of lane)
                        l_bound = boundaries[k+1]
                        r_bound = boundaries[k]
                        
                        msg_id = lane_ids[k]
                        msg_type = lane_types[k]
                        pred_id, succ_id = lane_links[k]
                        
                        # Pass road_id, links to Lane
                        new_lanes.append(Lane(msg_id, msg_type, l_bound, r_bound, is_junction, road_id, pred_id, succ_id))
                        # Attach road connectivity info to Lane as well? 
                        # Ideally Lane shouldn't hold this heavy info, but for simplicity:
                        new_lanes[-1].road_predecessor = road_predecessor
                        new_lanes[-1].road_successor = road_successor
                        
                        # Create Outer Line
                        m_type, m_color = lane_marks[k]
                        if m_type != "none":
                            new_lines.append(RoadLine(boundaries[k+1], m_type, m_color, is_junction))
                        elif is_junction and msg_type == 'driving':
                            # Force a virtual line for driving lanes in junctions if missing
                             new_lines.append(RoadLine(boundaries[k+1], 'none', 'white', is_junction))
                            
                    # Add Center Line if requested (once)
                    if add_center_line:
                         if center_lane_mark_type != "none":
                             new_lines.append(RoadLine(boundaries[0], center_lane_mark_type, center_lane_color, is_junction))
                         elif is_junction:
                             # Force center line in junction
                             new_lines.append(RoadLine(boundaries[0], 'none', 'white', is_junction))



                # Usually standard to add center line with Left side processing or just once
                process_side(ls.find('left'), 1, True)
                process_side(ls.find('right'), -1, False)
                
        self.lanes = new_lanes
        self.lines = new_lines
