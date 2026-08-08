import os
import torch
import numpy as np
import gymnasium as gym
import sys

# Add PPO_IBM to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PPO_IBM'))
from genetic_env import GeneticPhotobioreactorEnv
from dream_model import LightweightDreamerWorldModel, LightweightDreamerActor

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model_data/dreamer_light"
MAX_STEPS = 2000 # Longer to allow growth
NUM_EPISODES = 5

def main():
    print(f"--- Verifying Dreamer Agent on {DEVICE} ---")
    
    # 1. Initialize Env
    env = GeneticPhotobioreactorEnv(max_cells=300000, initial_cells=500)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Env Initialized. Obs: {obs_dim}, Action: {action_dim}")

    # 2. Load Models
    try:
        # Load World Model
        world_model = LightweightDreamerWorldModel(obs_dim, action_dim).to(DEVICE)
        wm_path = os.path.join(MODEL_DIR, "world_model.pth")
        world_model.load_state_dict(torch.load(wm_path, map_location=DEVICE))
        print(f"Loaded World Model from {wm_path}")
        
        # Load Actor
        actor = LightweightDreamerActor(feature_dim=384, action_dim=action_dim).to(DEVICE)
        actor_path = os.path.join(MODEL_DIR, "actor.pth")
        actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
        print(f"Loaded Actor from {actor_path}")
        
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 3. Run Episodes
    print(f"\n--- Starting Simulation ({NUM_EPISODES} Episodes) ---")
    
    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        print(f"\nEpisode {ep+1} | Strain: {env.strain_params}")
        
        # Initialize RSSM state
        rssm_state = world_model.rssm.initial_state(1, DEVICE)
        prev_action = torch.zeros(1, action_dim).to(DEVICE)
        
        total_reward = 0
        steps = 0
        
        while steps < MAX_STEPS:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                rssm_state, _ = world_model.rssm.observe_step(rssm_state, prev_action, obs_tensor)
                feature = world_model.rssm.get_features(rssm_state)
                action_dist = actor(feature)
                action = action_dist.cpu().numpy()[0]
                prev_action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            obs = next_obs
            
            if steps % 500 == 0:
                 print(f"  Step {steps:4d} | Reward: {reward:6.2f} | Pop: {info['pop']:6d} | OD: {obs[0]:.2f}")
            
            if done:
                break
                
        print(f"Episode {ep+1} Finished. Total Reward: {total_reward:.2f} | Steps: {steps}")
        
    print(f"\n--- Simulation Finished ---")

if __name__ == "__main__":
    main()
