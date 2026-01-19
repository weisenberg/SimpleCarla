import math
from .sensor_base import SensorBase

class ObstacleDetector(SensorBase):
    def __init__(self, parent_actor, world, distance=50, hit_radius=0.5, debug=False):
        super().__init__(parent_actor, world)
        self.distance = distance
        self.hit_radius = hit_radius
        self.closest_obstacle = None

    def tick(self, dt):
        parent = self._get_parent()
        if not parent: return
        
        self.closest_obstacle = None
        min_dist = self.distance
        
        if not hasattr(self.world, 'actors'): return
        
        px, py = parent.x, parent.y
        heading = getattr(parent, 'heading', 0)
        
        # Cone check
        for actor in self.world.actors:
            if actor.id == parent.id: continue
            
            dx = actor.x - px
            dy = actor.y - py
            dist = math.hypot(dx, dy)
            
            if dist > self.distance: continue
            
            # Angle check (Forward facing)
            angle = math.atan2(dy, dx) - heading
            angle = (angle + math.pi) % (2*math.pi) - math.pi
            
            if abs(angle) < math.radians(45): # 90 degree cone
                 if dist < min_dist:
                     min_dist = dist
                     self.closest_obstacle = actor
        
        if self.closest_obstacle and self.callback:
            self.callback({'distance': min_dist, 'actor': self.closest_obstacle})

    def visualize(self, surface):
        pass
