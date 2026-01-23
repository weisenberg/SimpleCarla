import gymnasium as gym
import sys
import os
import argparse
import time
from stable_baselines3 import PPO

# Add current directory to path
sys.path.append(os.getcwd())

from SimpleCarla.simple_carla_env import SimpleCarlaEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, required=False,
                       help="Path to model.zip (defaults to latest in SimpleCarla-logs)")
    args = parser.parse_args()

    # Find latest model if not provided
    model_path = args.model
    if not model_path:
        base_log_dir = "SimpleCarla-logs"
        if os.path.exists(base_log_dir):
            # Find latest timestamp folder
            runs = sorted(os.listdir(base_log_dir))
            if runs:
                latest_run = runs[-1]
                model_dir = os.path.join(base_log_dir, latest_run, "models")
                if os.path.exists(model_dir):
                    models = sorted(os.listdir(model_dir))
                    if models:
                        # Get best_model or latest checkpoint
                        model_path = os.path.join(model_dir, models[-1])
    
    if not model_path or not os.path.exists(model_path):
        print("❌ No model found! Please provide --model path or run training first.")
        return

    print(f"🎬 Loading Model: {model_path}")

    # Create Env in HUMAN mode
    sensors = {'lidar': True, 'collision': True, 'lane': True}
    env = SimpleCarlaEnv(render_mode="human", enable_ego=True, sensors=sensors, 
                        enable_traffic=True, traffic_density='mid', 
                        pedestrian_density='low', enable_pedestrians=True)

    # Load Model
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    
    print("🚗 Starting Visualization... (Press Ctrl+C to stop)")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                print(f"Episode Done. Reward: {env.total_reward:.2f}")
                obs, _ = env.reset()
            
            # Cap FPS for viewing
            time.sleep(0.05) 
            
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
