import math
import numpy as np
try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None

from .sensor_base import SensorBase

class CollisionSensor(SensorBase):
    def __init__(self, parent_actor, world):
        super().__init__(parent_actor, world)
        self.history = []

    def tick(self, dt):
        parent = self._get_parent()
        if not parent: return

        if Polygon is None:
            return # Cannot check without shapely for now

        # Create parent polygon
        parent_poly = self._get_actor_polygon(parent)
        
        # Iterate world actors
        # Assuming world.actors is a list of objects with (x, y, id)
        # and world.frame is current frame count
        if not hasattr(self.world, 'actors'):
            return

        for actor in self.world.actors:
            if actor.id == parent.id:
                continue

            # Optimization: Distance check
            dist = math.hypot(actor.x - parent.x, actor.y - parent.y)
            if dist > 10.0:
                continue

            # Check Intersection
            actor_poly = self._get_actor_polygon(actor)
            if parent_poly.intersects(actor_poly):
                # Calculate Impulse (Simplified: Difference in velocity * mass?)
                # Simplified impulse vector pointing from other to self
                impulse = np.array([parent.x - actor.x, parent.y - actor.y])
                norm = np.linalg.norm(impulse)
                if norm > 0: impulse /= norm
                
                event = {
                    'frame': getattr(self.world, 'frame', 0),
                    'other_actor': actor,
                    'impulse': impulse
                }
                
                if self.callback:
                    self.callback(event)
                
                # Debounce/History logic could go here to avoid spamming
                break

    def _get_actor_polygon(self, actor):
        # Create a rectangular polygon based on actor pose
        # Assuming actor.length, actor.width, actor.heading (rad)
        # x, y is center
        
        l2 = getattr(actor, 'length', 4.0) / 2.0
        w2 = getattr(actor, 'width', 2.0) / 2.0
        x = getattr(actor, 'x', 0)
        y = getattr(actor, 'y', 0)
        heading = getattr(actor, 'heading', 0)
        
        c = math.cos(heading)
        s = math.sin(heading)
        
        # Corners relative to center
        # FL, FR, RR, RL
        corners = [
            (l2, -w2), (l2, w2), (-l2, w2), (-l2, -w2)
        ]
        
        rotated_corners = []
        for cx, cy in corners:
            rx = x + (cx * c - cy * s)
            ry = y + (cx * s + cy * c)
            rotated_corners.append((rx, ry))
            
        return Polygon(rotated_corners)

    def visualize(self, surface):
        pass
