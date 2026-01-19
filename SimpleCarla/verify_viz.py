from simple_carla_env import SimpleCarlaEnv
from PIL import Image
import os
import sys

def verify(map_name):
    print(f"Verifying {map_name}...")
    # render_mode rgb_array to avoid window popup if possible, 
    # but my env implementation initializes window even for rgb_array if I am not careful?
    # My impl: if self.window is None and self.render_mode == "human": init window.
    # I should support "rgb_array" without window if possible, but Pygame needs a surface.
    # Pygame can draw to a surface without a window.
    
    # Let's fix env to allow headless rendering if mode is rgb_array?
    # My env:
    # canvas = pygame.Surface(self.window_size)
    # ... drawing on canvas ...
    # return np.transpose(...)
    # It does NOT verify window existence for drawing on canvas.
    # It ONLY inits window if mode == "human".
    # So "rgb_array" should work headless!
    
    env = SimpleCarlaEnv(map_name=map_name, render_mode="rgb_array")
    env.reset()
    frame = env.render()
    env.close()
    
    # Save frame
    img = Image.fromarray(frame)
    os.makedirs("SimpleCarla/viz_test", exist_ok=True)
    img.save(f"SimpleCarla/viz_test/{map_name}.png")
    print(f"Saved SimpleCarla/viz_test/{map_name}.png")

if __name__ == "__main__":
    maps = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
    for m in maps:
        try:
           verify(m)
        except Exception as e:
           print(f"Failed {m}: {e}")
