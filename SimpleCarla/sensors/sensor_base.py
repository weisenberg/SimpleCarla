import weakref
import pygame

class SensorBase:
    def __init__(self, parent_actor, world):
        self.parent = weakref.ref(parent_actor)
        self.world = world
        self.callback = None
    
    def listen(self, callback):
        """
        Register a callback function to be called when data is generated.
        callback(data)
        """
        self.callback = callback

    def stop(self):
        """Stop listening."""
        self.callback = None

    def destroy(self):
        """Cleanup resources."""
        self.stop()
        self.parent = None
        self.world = None

    def tick(self, dt):
        """
        Called every simulation step.
        Override this to generate data.
        """
        pass

    def visualize(self, surface):
        """
        Draw debug information on the Pygame surface.
        Override this to visualize sensor data.
        """
        pass

    def _get_parent(self):
        """Helper to get parent actor safely."""
        return self.parent()
