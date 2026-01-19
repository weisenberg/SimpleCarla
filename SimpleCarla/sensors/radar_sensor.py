import math
import numpy as np
import pygame
from .sensor_base import SensorBase

class RadarSensor(SensorBase):
    def __init__(self, parent_actor, world, fov=30, range=50):
        super().__init__(parent_actor, world)
        self.fov = math.radians(fov)
        self.range = range
        self.data = []

    def tick(self, dt):
        parent = self._get_parent()
        if not parent: return

        # Simulate radar detections
        # We need relative velocity.
        
        self.data = []
        if not hasattr(self.world, 'actors'): return

        px, py = parent.x, parent.y
        pvx = getattr(parent, 'velocity_x', 0)
        pvy = getattr(parent, 'velocity_y', 0)
        heading = getattr(parent, 'heading', 0)
        
        # FOV cone direction
        dir_x = math.cos(heading)
        dir_y = math.sin(heading)

        for actor in self.world.actors:
            if actor.id == parent.id: continue
            
            dx = actor.x - px
            dy = actor.y - py
            dist = math.hypot(dx, dy)
            
            if dist > self.range: continue
            
            # Check Angle
            angle_to_actor = math.atan2(dy, dx)
            angle_diff = angle_to_actor - heading
            # Normalize -pi to pi
            angle_diff = (angle_diff + math.pi) % (2*math.pi) - math.pi
            
            if abs(angle_diff) < self.fov / 2:
                # Detected
                # Relative Vel
                avx = getattr(actor, 'velocity_x', 0)
                avy = getattr(actor, 'velocity_y', 0)
                
                rel_vx = avx - pvx
                rel_vy = avy - pvy
                
                # Project onto ray direction (dx, dy normalized)
                ray_x, ray_y = dx/dist, dy/dist
                velocity = rel_vx * ray_x + rel_vy * ray_y
                
                azimuth = angle_diff
                altitude = 0
                depth = dist
                
                self.data.append([velocity, azimuth, altitude, depth])
        
        if self.callback:
            self.callback(np.array(self.data))

    def visualize(self, surface):
        parent = self._get_parent()
        if not parent: return
        
        # Draw FOV cone
        # ...
        pass
