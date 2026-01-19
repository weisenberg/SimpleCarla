from scenario_map import OpenDriveParser

def main():
    map_path = "Maps/CARLA/Town01.xodr"
    parser = OpenDriveParser(map_path)
    parser.parse()
    
    print(f"Total Lanes: {len(parser.lanes)}")
    print(f"Total Lines: {len(parser.lines)}")
    
    # Count types
    types = {}
    colors = {}
    for l in parser.lines:
        types[l.type] = types.get(l.type, 0) + 1
        colors[l.color] = colors.get(l.color, 0) + 1
        
    print("Line Types:", types)
    print("Line Colors:", colors)
    
    # Check if we have 'broken' lines (center)
    has_broken = any(l.type == 'broken' for l in parser.lines)
    print(f"Has broken lines? {has_broken}")

if __name__ == "__main__":
    main()
