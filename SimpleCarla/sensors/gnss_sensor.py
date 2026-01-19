import math
from .sensor_base import SensorBase

class GnssSensor(SensorBase):
    def __init__(self, parent_actor, world):
        super().__init__(parent_actor, world)
        # Reference coords (e.g., CARLA default Town01)
        self.initial_lat = 0.0
        self.initial_lon = 0.0

    def tick(self, dt):
        parent = self._get_parent()
        if not parent: return
        
        x = parent.x
        y = parent.y
        
        # Math from prompt
        # lat = initial_lat + (y / 111320.0)
        # lon = initial_lon + (x / (111320.0 * cos(lat)))
        
        lat = self.initial_lat + (y / 111320.0)
        try:
            lon = self.initial_lon + (x / (111320.0 * math.cos(math.radians(lat))))
        except ZeroDivisionError:
            lon = self.initial_lon
            
        data = {'lat': lat, 'lon': lon, 'alt': 0.0}
        
        if self.callback:
            self.callback(data)
