import gymnasium as gym
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Fix path for SubprocVecEnv workers
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv, VecEnvWrapper
from simple_carla_env import SimpleCarlaEnv

class SingleCoreRenderWrapper(VecEnvWrapper):
    """
    Wrapper that forces render() to only return the image from the first environment (Rank 0),
    bypassing the default tiling of all environments.
    """
    def __init__(self, venv):
        super().__init__(venv)
    
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def render(self, mode='human'):
        # Only ask Core 0 to render
        # We assume self.venv is a SubprocVecEnv which has 'remotes'
        if hasattr(self.venv, 'remotes'):
             self.venv.remotes[0].send(('render', mode))
             img = self.venv.remotes[0].recv()
             return img
        else:
             # Fallback for DummyVecEnv or others
             imgs = self.venv.get_images()
             return imgs[0] if len(imgs) > 0 else None

class DomainRandomizationCallback(BaseCallback):
    """
    Callback for domain randomization (resampling Traffic/Pedestrian config)
    at the start of each episode.
    """
    def __init__(self, verbose=0):
        super(DomainRandomizationCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        # Randomize Traffic/Peds for the next episodes
        # With VecEnv, we can use set_attr to broadcast to all workers.
        # This will take effect next time an env resets.
        
        traffic_modes = ['no', 'low', 'mid', 'high']
        peds_modes = ['no', 'low', 'mid', 'high']
        
        t = np.random.choice(traffic_modes)
        p = np.random.choice(peds_modes)
        
        try:
            # Broadcast to all parallel environments
            self.training_env.set_attr("traffic_density", t)
            self.training_env.set_attr("enable_traffic", (t != 'no'))
            
            self.training_env.set_attr("pedestrian_density", p)
            self.training_env.set_attr("enable_pedestrians", (p != 'no'))
            
            if self.verbose > 0:
                pass # Too noisy to print every rollout with 4 workers
        except Exception as e:
            if self.verbose > 0:
                print(f"Domain Randomization Failed: {e}")

class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CSVLoggerCallback, self).__init__(verbose)
        self.log_path = os.path.join(log_dir, "episode_log.csv")
        self.episode_count = 0
        self.header_written = False
        
        # Create file
        with open(self.log_path, 'w') as f:
            f.write("episode,reward,reason,duration,traffic,pedestrians,map\n")

    def _on_step(self) -> bool:
        # SB3 Monitor wrapper logs to .monitor.csv, but we want custom "Reason".
        # We check locals for 'infos'
        infos = self.locals.get("infos", [{}])[0]
        
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            reason = "Unknown"
            if infos.get("TimeLimit.truncated", False):
                reason = "TimeLimit"
            elif infos.get("is_collision", False):
                reason = "Collision"
            elif infos.get("is_off_road", False): # Approximation
                reason = "OffRoad"
            else:
                 # Check latent info?
                 reason = "Other"
            
            # Env state
            # Compatible with SubprocVecEnv using get_attr
            try:
                # Fetch from the first environment (index 0)
                t_density = self.training_env.get_attr("traffic_density", indices=[0])[0]
                t_enabled = self.training_env.get_attr("enable_traffic", indices=[0])[0]
                t = t_density if t_enabled else "no"
                
                p_density = self.training_env.get_attr("pedestrian_density", indices=[0])[0]
                p_enabled = self.training_env.get_attr("enable_pedestrians", indices=[0])[0]
                p = p_density if p_enabled else "no"
                
                m = self.training_env.get_attr("map_name", indices=[0])[0]
            except Exception:
                # Fallback if get_attr fails or env structure differs
                t, p, m = "unknown", "unknown", "unknown"
            
            # Reward is trickier to get cumulative from here without Monitor, 
            # but Monitor puts it in info['episode']['r']
            ep_reward = 0.0
            ep_len = 0.0
            if 'episode' in infos:
                ep_reward = infos['episode']['r']
                ep_len = infos['episode']['l'] # Steps?
                # Duration in seconds = steps / 30
                duration = ep_len / 30.0
            
            with open(self.log_path, 'a') as f:
                  f.write(f"{self.episode_count},{ep_reward},{reason},{duration:.2f},{t},{p},{m}\n")
            
            # Plotting 
            self._update_plot()
                  
        return True

    def _update_plot(self):
        try:
            # Read data (skipping header) - efficient enough for <10k lines
            # If slow, we can cache data in memory
            data = np.genfromtxt(self.log_path, delimiter=',', skip_header=1, usecols=(1,))
            if data.ndim == 0: data = np.array([data]) # Handle single scalar
            
            if len(data) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(data, label='Episode Reward', alpha=0.6, color='blue')
                
                # Rolling Average (N=50)
                if len(data) >= 50:
                    # Convolve for moving average
                    weights = np.ones(50) / 50
                    # mode='valid' starts only when we have enough data
                    ma = np.convolve(data, weights, mode='valid')
                    # x-axis adjustment: MA starts at index 49 (50th item)
                    plt.plot(np.arange(49, 49 + len(ma)), ma, color='red', linewidth=2, label='Avg (50)')
                
                plt.title("Reward per Episode")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_path = os.path.join(os.path.dirname(self.log_path), "training_plot.png")
                plt.savefig(plot_path)
                plt.close()
        except Exception as e:
            if self.verbose > 0:
                print(f"Plotting error: {e}")

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value
    return func

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, StopTrainingOnRewardThreshold, EvalCallback


class RobustConvergenceCallback(BaseCallback):
    """
    Robust stopping criteria based on rolling average and stability.
    
    Stops training when:
    1. Mean reward (last N episodes) > threshold
    2. Std dev (last N episodes) < max_variance (ensures stability)
    3. At least min_episodes have been completed
    """
    def __init__(self, reward_threshold: float = 500.0, window_size: int = 100, 
                 min_episodes: int = 200, max_variance: float = 100.0, verbose=1):
        super(RobustConvergenceCallback, self).__init__(verbose)
        self.reward_threshold = reward_threshold
        self.window_size = window_size
        self.min_episodes = min_episodes
        self.max_variance = max_variance
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones", [False])[0]:
            infos = self.locals.get("infos", [{}])[0]
            if 'episode' in infos:
                ep_reward = infos['episode']['r']
                self.episode_rewards.append(ep_reward)
                self.episode_count += 1
                
                # Keep only last window_size episodes
                if len(self.episode_rewards) > self.window_size:
                    self.episode_rewards = self.episode_rewards[-self.window_size:]
                
                # Check convergence criteria
                if self.episode_count >= self.min_episodes and len(self.episode_rewards) >= self.window_size:
                    mean_reward = np.mean(self.episode_rewards)
                    std_reward = np.std(self.episode_rewards)
                    
                    if mean_reward > self.reward_threshold and std_reward < self.max_variance:
                        if self.verbose > 0:
                            print(f"\nStopping training: Robust convergence achieved!")
                            print(f"  Mean Reward (last {self.window_size} eps): {mean_reward:.2f}")
                            print(f"  Std Dev: {std_reward:.2f}")
                            print(f"  Episodes Completed: {self.episode_count}")
                        return False
        
        return True


def make_env():
    # Helper to create env (Gym <-> Gymnasium compat handled by Shimmy if installed, else we assume Gym API)
    # SimpleCarlaEnv is gymnasium.Env
    sensors = {'lidar': True, 'collision': True, 'lane': True}
    # Default to 'mid' traffic, but DomainRandomizationCallback will override this on rollout start
    # RENDER_MODE='rgb_array' allows VecVideoRecorder to work. 
    # Since render() is only called when recording, it won't slow down normal steps significantly.
    env = SimpleCarlaEnv(render_mode="rgb_array", enable_ego=True, sensors=sensors, 
                        enable_traffic=True, traffic_density='mid', 
                        pedestrian_density='low', enable_pedestrians=True) 
    
    # Apply Wrapper
    from wrappers import CarlaObservationWrapper
    env = CarlaObservationWrapper(env)
    
    env = Monitor(env) # For SB3 logging
    return env

from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv

def make_env_thunk(rank, seed=0):
    """
    Utility to return a no-argument function for SubprocVecEnv.
    """
    def _init():
        env = make_env()
        # Seed logic could be here
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    # Default to 1M steps for standard training
    parser.add_argument("--steps", type=int, default=1000000, 
                       help="Total training timesteps")
    args = parser.parse_args()

    # Logging Setup
    base_dir = "/Users/ali/Desktop/uni/master thesis/playground/SimpleCarla-logs"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(base_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "videos"), exist_ok=True)
    
    print(f"Logging to: {log_dir}")
    
    # Environment Setup: Parallel Training
    # Use SubprocVecEnv for 8 parallel processes (Speed boost)
    num_cpu = 8  
    # Create the vectorized environment
    env = SubprocVecEnv([make_env_thunk(i) for i in range(num_cpu)])
    
    # WRAPPER TO FIX VIDEO GRID
    # Ensures get_images/render only returns Core 0 (Single Video, No Grid)
    env = SingleCoreRenderWrapper(env)
    
    # Video Recorder Wrapper
    # Record approximately every 100 global episodes.
    # 8 Environments. 1 Episode ~ 500 steps. 
    # To save every ~100 episodes total: 100 / 8 = 12 episodes per env.
    # 12 * 500 = 6000 steps.
    video_folder = os.path.join(log_dir, "videos")
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x % 6000 == 0, 
                           video_length=700, # Capture exactly one episode (30s * 20fps = 600)
                           name_prefix=f"agent-ppo")

    # Callbacks
    # DomainRandomizationCallback needs to handle VecEnv?
    # Our simple callback tries to access training_env.envs[0].
    # With SubprocVecEnv, we can't easily access the remote envs directly to change attributes.
    # We might drop DomainRandomizationCallback for now or rely on random resets inside env.
    # SimpleCarlaEnv DOES randomize on reset if not passed options? No, it defaults to 'no'.
    # We must fix this if we want randomization.
    
    # CSVLoggerCallback also expects local access.
    # This is a limitation of SubprocVecEnv + Custom Callbacks logging "reason".
    # We will wrap the SubprocVecEnv with a VecMonitor that logs to a file, 
    # OR we use the CSVLogger but it will only log from the main process if info is passed back.
    # SubprocVecEnv passes info back.
    
    csv_cb = CSVLoggerCallback(log_dir)
    checkpoint_cb = CheckpointCallback(save_freq=10000 // num_cpu, save_path=os.path.join(log_dir, "models"), name_prefix="ppo_carla")
    
    # Model
    # Alpha Decay: 3e-4 to 0
    lr_schedule = linear_schedule(3e-4)
    
    
    # Robust Convergence Callback (Instead of EvalCallback with StopTrainingOnRewardThreshold)
    # This uses rolling average over last 100 episodes from training (not just 5 eval episodes)
    robust_stop_cb = RobustConvergenceCallback(
        reward_threshold=1000.0,  # Achievable with ~20s of good driving
        window_size=100,           # Rolling average over 100 episodes
        min_episodes=200,          # Must complete at least 200 episodes
        max_variance=200.0,        # Std dev must be < 200 (ensures stability)
        verbose=1
    )

    # Tuning PPO:
    # ent_coef=0.05: Force more exploration (prevent "scared stopping").
    # n_steps=1024: Collection size per env
    # batch_size=64: Opt size
    model = PPO("MultiInputPolicy", env, learning_rate=lr_schedule, verbose=1, 
                tensorboard_log=os.path.join(log_dir, "tensorboard"),
                ent_coef=0.05,
                n_steps=1024,
                batch_size=64)
    
    print(f"Starting Training on {num_cpu} CPUs...")
    
    # Callbacks list
    domain_cb = DomainRandomizationCallback() # Re-instantiate
    callbacks = [domain_cb, csv_cb, checkpoint_cb, robust_stop_cb]
    
    model.learn(total_timesteps=args.steps, callback=callbacks)
    
    # Save Final
    model.save(os.path.join(log_dir, "models", "final_model"))
    print("Training Complete. Model Saved.")
    
    # Plotting (Simple Reward Plot from CSV)
    try:
        data = np.genfromtxt(os.path.join(log_dir, "episode_log.csv"), delimiter=',', skip_header=1, usecols=(1,))
        plt.figure()
        plt.plot(data)
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(log_dir, "training_plot.png"))
        print("Plot saved.")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
