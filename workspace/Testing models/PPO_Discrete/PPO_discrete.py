import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import torch
import torch.nn as nn
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import PhotobioreactorDiscreteEnv
from reward_functions import default_reward_function

"""
PPO_discrete.py

This file implements a PPO agent for the Discrete Action Space version of the PBR.
It uses a trained World Model (Supervisor) to simulate environment dynamics based on CSV data.

STATE (Observation):
- Microalgae Density (float), normalized [0, 1]

ACTION (Controls) - 4 Discrete Actuators, 5 levels each (0-4):
- a1: Stirring Rate
- a2: Solution Amount
- a3: Rate of Flow
- a4: Light Intensity
"""

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Initializing Discrete Photobioreactor Environment with World Model...")
    
    # 1. Create the environment
    # We can pass a custom reward function here if we want!
    # e.g., env = DummyVecEnv([lambda: PhotobioreactorDiscreteEnv(reward_func=my_custom_reward)])
    env = DummyVecEnv([lambda: PhotobioreactorDiscreteEnv()])
    print("Environment created.")

    # 2. Define or Load the PPO model
    model_path = "ppo_discrete_pbr_wm.zip"
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            # We need to pass env so it can continue training
            model = PPO.load(model_path, env=env)
            print("Model loaded. Resuming training...")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            print("Backing up corrupt model and starting fresh...")
            os.rename(model_path, model_path + ".bak")
            
            print("Creating new PPO model...")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log="./ppo_discrete_tensorboard/",
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                learning_rate=3e-4
            )
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_discrete_tensorboard/",
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            learning_rate=3e-4
        )
    
    # 3. Train
    print("Starting model training (Discrete + World Model)...")
    model.learn(total_timesteps=15000, progress_bar=True, reset_num_timesteps=False)
    print("Training complete.")
    
    model.save("ppo_discrete_pbr_wm")
    print("Model saved to ppo_discrete_pbr_wm.zip")
    
    # 4. Evaluate
    print("\n--- Evaluating Using World Model Dynamics ---")
    
    obs = env.reset()
    total_reward = 0
    
    # We want to see if the agent exploits the learned dynamics
    for step in range(20): 
        action, _states = model.predict(obs, deterministic=True)
        act = action[0] 
        
        print(f"Step {step}:")
        print(f"  Obs (Density): {obs[0][0]:.4f}")
        print(f"  Action -> Stir:{act[0]}, Sol:{act[1]}, Flow:{act[2]}, Light:{act[3]}")
        
        obs, reward, terminated, info = env.step(action)
        total_reward += reward[0]
        
        print(f"  Result Density: {info[0]['density']:.4f}, Reward: {reward[0]:.4f}")
        
    print(f"Total Reward (10 steps): {total_reward:.4f}")
    env.close()
