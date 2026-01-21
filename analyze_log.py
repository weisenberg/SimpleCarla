
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Path to the latest log based on previous step
latest_log_dir = "/Users/ali/Desktop/uni/master thesis/playground/SimpleCarla-logs/2026-01-21_13-01-38"
log_path = os.path.join(latest_log_dir, 'episode_log.csv')

print(f"Analyzing log: {log_path}")

try:
    if not os.path.exists(log_path):
        print("Error: Log file not found.")
        sys.exit(1)

    df = pd.read_csv(log_path)
    print(f"Loaded {len(df)} rows.")
    
    # --- 1. Average Episode Length ---
    # In our log, 'duration' is in seconds.
    # Steps = duration * 30 (FPS)
    if 'duration' in df.columns:
        avg_dur = df['duration'].mean()
        avg_steps = avg_dur * 30
        print(f"Average Episode Length: {avg_steps:.1f} steps ({avg_dur:.1f} seconds)")
    else:
        print("Column 'duration' not found.")

    # --- 2. Termination Reasons ---
    # Our log has a 'reason' column directly
    if 'reason' in df.columns:
        counts = df['reason'].value_counts()
        total = len(df)
        print("Terminations:")
        for reason, count in counts.items():
            print(f"  - {reason}: {count} ({count/total*100:.1f}%)")
            
        # Also check 'TimeLimit' logic explicitly if needed
        # But 'reason' logic in callback already handles this map:
        # "TimeLimit" -> Truncated
        # "Collision" -> Terminated
        # "OffRoad" -> Terminated
    else:
        print("Column 'reason' not found.")

    # --- 3. Steering Noise ---
    # Our CSVLoggerCallback does NOT log actions (steer/throttle).
    # We cannot calculate steering std dev from this summary log.
    print("Steering Stability: [N/A] (Steering actions not logged in episode_log.csv)")
    
    # --- Extra: Traffic Distribution ---
    if 'traffic' in df.columns:
        print("\nTraffic Settings Distribution:")
        print(df['traffic'].value_counts(normalize=True))

except Exception as e:
    print(f"Could not parse log: {e}")
