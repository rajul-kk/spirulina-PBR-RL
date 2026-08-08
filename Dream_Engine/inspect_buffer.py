import sys
import os
import pickle
import numpy as np

# Config
MODEL_DIR = "model_data/dreamer_light"
BUFFER_PATH = os.path.join(MODEL_DIR, "buffer.pkl")

def main():
    print(f"--- Inspecting Replay Buffer ---")
    
    if not os.path.exists(BUFFER_PATH):
        print(f"Buffer file not found: {BUFFER_PATH}")
        return

    try:
        with open(BUFFER_PATH, 'rb') as f:
            episodes = pickle.load(f)
            
        print(f"Loaded ReplayBuffer with {len(episodes)} episodes.")
        
        all_rewards = []
        episode_rewards = []
        
        for i, ep in enumerate(episodes):
            rewards = ep['reward']
            all_rewards.append(rewards)
            total = np.sum(rewards)
            episode_rewards.append(total)
            
            if i < 5: # Print first 5
                print(f"  Ep {i+1}: Length {len(rewards)}, Total Reward: {total:.4f}")
                
        all_rewards = np.concatenate(all_rewards)
        episode_rewards = np.array(episode_rewards)
        
        print("\n--- Statistics ---")
        print(f"Total Steps Stored: {len(all_rewards)}")
        print(f"Reward Mean: {np.mean(all_rewards):.6f}")
        print(f"Reward Std:  {np.std(all_rewards):.6f}")
        print(f"Reward Min:  {np.min(all_rewards):.6f}")
        print(f"Reward Max:  {np.max(all_rewards):.6f}")
        
        print(f"\nEpisode Return Mean: {np.mean(episode_rewards):.4f}")
        print(f"Episode Return Max:  {np.max(episode_rewards):.4f}")
        
        # Check non-zero rewards
        non_zero = all_rewards[np.abs(all_rewards) > 0.001]
        print(f"\nNon-zero Rewards (>0.001): {len(non_zero)} / {len(all_rewards)}")
        
    except Exception as e:
        print(f"Error reading buffer: {e}")

if __name__ == "__main__":
    main()
