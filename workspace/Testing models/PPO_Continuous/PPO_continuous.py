import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from experiment_env import PhotobioreactorExperimentEnv

"""
PPO_continuous.py

This file implements a PPO agent for the Continuous Action Space version of the PBR.
It uses the advanced Huisman/Monod hybrid physics model.

STATE (Observation):
- [Water, Light_In, pH, Temp, Nutrients, OD]

ACTION (Continuous [-1, 1]):
- Light Setpoint
- Temp Setpoint
- CO2 Flow
- Nutrient Flow
"""

def main():
    print("Initializing Continuous Photobioreactor Environment...")
    
    # 1. Create Environment
    env = DummyVecEnv([lambda: PhotobioreactorExperimentEnv()])
    
    # 2. Define Model
    model_path = "ppo_continuous_pbr.zip"
    
    # Check if we can resume
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            model = PPO.load(model_path, env=env)
            print("Model loaded. Resuming training...")
        except Exception as e:
            print(f"Error loading model: {e}. Creating new one.")
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_continuous_tensorboard/")
    else:
        print("Creating new PPO model (MlpPolicy)...")
        # Continuous PPO uses Gaussian distribution for actions by default with MlpPolicy
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log="./ppo_continuous_tensorboard/",
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99
        )
        
    # 3. Train
    print("Starting training...")
    try:
        model.learn(total_timesteps=50000, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
        
    # 4. Save
    model.save("ppo_continuous_pbr")
    print("Model saved.")
    
    # 5. Evaluate
    print("\n--- Evaluation ---")
    obs = env.reset()
    total_prod = 0
    
    for i in range(24): # 1 day
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        prod = info[0]['productivity']
        total_prod += prod
        env.envs[0].render()
        
        if done:
            break
            
    print(f"Total Productivity (24h): {total_prod:.4f} g/L")

if __name__ == "__main__":
    main()
