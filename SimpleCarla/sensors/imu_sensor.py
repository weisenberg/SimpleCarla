import math
import random
from .sensor_base import SensorBase

class ImuSensor(SensorBase):
    def __init__(self, parent_actor, world):
        super().__init__(parent_actor, world)
        self.last_velocity = (0, 0)
        self.last_yaw = 0
        self.accelerometer = (0, 0, 0)
        self.gyroscope = (0, 0, 0)
        self.compass = 0 

    def tick(self, dt):
        parent = self._get_parent()
        if not parent: return
        if dt <= 0: return
        
        vx = getattr(parent, 'velocity_x', 0)
        vy = getattr(parent, 'velocity_y', 0)
        yaw = getattr(parent, 'heading', 0)
        
        # Calculate accel
        dvx = vx - self.last_velocity[0]
        dvy = vy - self.last_velocity[1]
        
        acc_x = dvx / dt
        acc_y = dvy / dt
        # Gravity? typically IMU includes gravity component (9.81 up/down)
        # Here 2D.
        
        # Calculate gyro
        dyaw = yaw - self.last_yaw
        dyaw = (dyaw + math.pi) % (2*math.pi) - math.pi # Wrap
        gyro_z = dyaw / dt
        
        # Noise
        noise_acc = random.gauss(0, 0.1)
        noise_gyro = random.gauss(0, 0.05)
        
        self.accelerometer = (acc_x + noise_acc, acc_y + noise_acc, 9.81)
        self.gyroscope = (0, 0, gyro_z + noise_gyro)
        self.compass = yaw
        
        self.last_velocity = (vx, vy)
        self.last_yaw = yaw
        
        data = {
            'accelerometer': self.accelerometer,
            'gyroscope': self.gyroscope,
            'compass': self.compass
        }
        
        if self.callback:
            self.callback(data)
