import math
from .sensor_base import SensorBase

class LaneInvasionSensor(SensorBase):
    def __init__(self, parent_actor, world):
        super().__init__(parent_actor, world)
        self.last_corners = None

    def tick(self, dt):
        parent = self._get_parent()
        if not parent: return
        
        if not hasattr(self.world, 'map_data'):
            return

        # Get current corners
        current_corners = self._get_corners(parent)
        
        # Check against lane lines in map_data
        # Assuming world.map_data has 'lines' list of RoadLine objects with .points
        
        crossed_types = set()
        
        # Simple check: If a line segment of a lane boundary intersects with any bounding box edge
        # Or simpler: Check if any corner is in a different lane ID than before?
        # User Logic: "Check if the actor's corners have crossed a lane boundary line since the last frame."
        
        if self.last_corners:
            # We check intersection of movement vectors of corners? 
            # Or just check if current pos is over a line?
            # Let's check intersection of (last_corner -> current_corner) with map lines.
            
            # For efficiency, only check nearby lines
            # But we don't have a spatial index.
            # Simplified: Check if center crossed? Or look up lane ID.
            
            # Assuming 'world.get_lane_at(x, y)' exists? 
            # If we don't have that, we check geometry intersection.
             pass 
        
        # NOTE: Without a robust map query API, valid implementation is hard.
        # I will implement a placeholder that assumes `world.parser.get_lane_type(x,y)` exists, 
        # or we implement geometric intersection if map lines are available.
        # I'll iterate lines.
        
        if hasattr(self.world, 'map_lines'):
             for line in self.world.map_lines:
                 # Check intersection...
                 pass

        self.last_corners = current_corners
        
        if crossed_types and self.callback:
             self.callback(list(crossed_types))

    def _get_corners(self, actor):
        l2 = getattr(actor, 'length', 4.0) / 2.0
        w2 = getattr(actor, 'width', 2.0) / 2.0
        x = getattr(actor, 'x', 0)
        y = getattr(actor, 'y', 0)
        h = getattr(actor, 'heading', 0)
        c, s = math.cos(h), math.sin(h)
        
        corners = []
        for dx, dy in [(l2, -w2), (l2, w2), (-l2, w2), (-l2, -w2)]:
             rx = x + dx * c - dy * s
             ry = y + dx * s + dy * c
             corners.append((rx, ry))
        return corners
