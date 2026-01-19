import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import math
import glob

def rot_matrix(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])

class OpenDriveParser:
    def __init__(self, map_file):
        self.tree = ET.parse(map_file)
        self.root = self.tree.getroot()
        self.points = [] # list of (x, y) for reference line
        self.lanes = []  # list of lists of (x, y) for lane boundaries

    def get_width(self, width_objs, ds):
        # width_objs is list of parsed <width> elements: {sOffset, a, b, c, d}
        # Find the active width record
        # Note: XML 'width' children of 'lane' usually are sorted by sOffset
        # We need the one where sOffset <= ds < next_sOffset
        
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
        for road in self.root.findall('road'):
            # 1. Get reference geometry
            # Often multiple geometry records per planView
            plan_view = road.find('planView')
            geometries = plan_view.findall('geometry')
            
            # Sort by 's' just in case
            geometries.sort(key=lambda g: float(g.get('s')))
            
            # Pre-parse geometry ranges
            geo_ranges = []
            for g in geometries:
                s = float(g.get('s'))
                l = float(g.get('length'))
                geo_ranges.append({'s': s, 'end': s + l, 'elem': g})
                
            lanes = road.find('lanes')
            if lanes is None: continue
            
            lane_sections = lanes.findall('laneSection')
            lane_sections.sort(key=lambda ls: float(ls.get('s')))
            
            # Determine end of road for last section
            road_length = float(road.get('length'))
            
            for i, ls in enumerate(lane_sections):
                s_section = float(ls.get('s'))
                if i < len(lane_sections) - 1:
                    s_section_end = float(lane_sections[i+1].get('s'))
                else:
                    s_section_end = road_length
                    
                # Parse lanes in this section
                # Left (pos ID), Center (0), Right (neg ID)
                # Need to track accumulated width from center
                
                # Helper to process side
                def process_side(side_node, side_sign): # side_sign: 1 for left, -1 for right? 
                    # Actually standard: left is +id, right is -id.
                    # Width accumulation: start at center (offset 0).
                    # Left: boundary moves along +normal (left).
                    # Right: boundary moves along -normal (right).
                    
                    if side_node is None: return
                    
                    # Sort lanes by abs(id) ascending (closest to center first)
                    ls_lanes = side_node.findall('lane')
                    ls_lanes.sort(key=lambda l: abs(int(l.get('id'))))
                    
                    # We only care about boundaries.
                    # Lane n matches Space between Boundary n-1 and Boundary n.
                    # So we track 'current_offset' from reference line.
                    
                    current_width_funcs = [] # List of width objs per lane
                    
                    for lane in ls_lanes:
                        l_id = int(lane.get('id'))
                        # parse width elements
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
                        current_width_funcs.append({'id': l_id, 'widths': w_objs})

                    # Now iterate S over this section
                    # We just sample points
                    step = 1.0
                    n_steps = int((s_section_end - s_section) / step)
                    
                    # Store point lists for each lane boundary
                    # boundary_lines[k] corresponds to boundary after k-th lane
                    boundary_lines = [ [] for _ in range(len(ls_lanes)) ] 
                    
                    for j in range(n_steps + 1):
                        s_curr =  min(s_section + j * step, s_section_end)
                        
                        # Find geometry
                        geo = None
                        for gr in geo_ranges:
                            if gr['s'] <= s_curr <= gr['end'] + 0.1: 
                                geo = gr
                                break
                        if geo is None: continue 
                        
                        # Calculate ref pose at s_curr
                        ds_geo = s_curr - geo['s']
                        g_elem = geo['elem']
                        
                        x0 = float(g_elem.get('x'))
                        y0 = float(g_elem.get('y'))
                        h0 = float(g_elem.get('hdg'))
                        
                        rx, ry, rh = x0, y0, h0

                        if g_elem.find('line') is not None:
                            rx = x0 + ds_geo * math.cos(h0)
                            ry = y0 + ds_geo * math.sin(h0)
                            rh = h0
                        elif g_elem.find('arc') is not None:
                            c = float(g_elem.find('arc').get('curvature'))
                            rh = h0 + c * ds_geo
                            # Prevent div by zero if very straight arc
                            if abs(c) > 1e-10:
                                rx = x0 + (math.sin(rh) - math.sin(h0))/c
                                ry = y0 + (math.cos(h0) - math.cos(rh))/c
                            else:
                                rx = x0 + ds_geo * math.cos(h0)
                                ry = y0 + ds_geo * math.sin(h0)
                        elif g_elem.find('spiral') is not None:
                            # Basic Euler for spiral fallback
                            # Ideally should integrate properly from 0 to ds_geo
                            # But since we are stepping through the ROAD S (s_curr), 
                            # we can actually integrate incrementally if we processed road sequentially.
                            # But here we jump around.
                            # For visualization, linear interp might be 'okay' if spiral is short, but let's do mini-integration
                            
                            curv_start = float(g_elem.find('spiral').get('curvStart'))
                            curv_end = float(g_elem.find('spiral').get('curvEnd'))
                            length = float(g_elem.get('length'))
                            c_dot = (curv_end - curv_start) / length
                            
                            # Simpson's rule or just small step integration for this single point? 
                            # Small step integration from start of geometry
                            # 10 steps max
                            n_sub = 10
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
                            # Fallback generic
                            rx = x0 + ds_geo * math.cos(h0)
                            ry = y0 + ds_geo * math.sin(h0)
                            rh = h0

                        # Calculate normal direction (left: +90 deg relative to heading)
                        nx = -math.sin(rh)
                        ny = math.cos(rh)
                        
                        accum_width = 0.0
                        
                        # Iterate lanes moving outwards
                        for k, l_data in enumerate(current_width_funcs):
                            w = self.get_width(l_data['widths'], s_curr - s_section)
                            accum_width += w
                            
                            eff_width = accum_width if side_sign > 0 else -accum_width
                            
                            bx = rx + nx * eff_width
                            by = ry + ny * eff_width
                            
                            boundary_lines[k].append((bx, by))
                            
                    self.lanes.extend(boundary_lines)

                process_side(ls.find('left'), 1)
                process_side(ls.find('right'), -1)

def main():
    map_dir = "/Users/ali/Desktop/uni/master thesis/playground/Carla/Maps/CARLA"
    # Need to output to SimpleCarla which is in the parent dir of Maps/CARLA wait...
    # /Users/ali/Desktop/uni/master thesis/playground/Carla/Maps/CARLA is where maps are.
    # User said save in "parallel dir named SimpleCarla".
    # Assuming parallel to Maps? Or parallel to Maps/CARLA? 
    # USER -> "at /Maps/CARLA ... Save in parallel dir named SimpleCarla"
    # Usually means /Maps/SimpleCarla OR /SimpleCarla depending on root.
    # Given workspace: /Users/ali/Desktop/uni/master thesis/playground/Carla
    # Relative path /Maps/CARLA exists.
    # I already created SimpleCarla in the root workspace. I will use that.
    
    out_dir = "/Users/ali/Desktop/uni/master thesis/playground/Carla/SimpleCarla/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(map_dir, "*.xodr"))
    
    for f in files:
        if "Opt" in f: continue
        
        name = os.path.basename(f).replace('.xodr', '')
        print(f"Parsing {name}...")
        
        parser = OpenDriveParser(f)
        parser.parse()
        
        plt.figure(figsize=(12, 12))
        for lane in parser.lanes:
            if len(lane) < 2: continue
            arr = np.array(lane)
            plt.plot(arr[:,0], arr[:,1], linewidth=0.5, color='black')
        
        plt.axis('equal')
        plt.title(name)
        plt.savefig(os.path.join(out_dir, f"{name}.png"))
        plt.close()
        print(f"Saved plot to {out_dir}/{name}.png")

if __name__ == "__main__":
    main()
