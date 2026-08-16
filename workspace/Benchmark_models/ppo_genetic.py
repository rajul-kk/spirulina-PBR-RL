
import gymnasium as gym
import numpy as np
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import pickle

from genetic_env import GeneticPhotobioreactorEnv

class GeneticMonitorCallback(BaseCallback):
    """
    Custom callback to log the Genetic Parameters of each episode.
    """
    def __init__(self, verbose=0):
        super(GeneticMonitorCallback, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Detect new episodes to log the new random strain
        dones = self.locals['dones']
        if dones[0]:
            self.episode_count += 1
            env = self.training_env.envs[0]
            params = env.strain_params
            
            summary = f"Episode {self.episode_count} Finished. Next Strain: " \
                      f"Mu={params['mu_max']:.2f}, Ki={params['Ki']:.0f}"
            print(f"[GeneticMonitor] {summary}")
            
        return True

def main():
    print("--- PPO Agent for Genetic IBM Photobioreactor ---")
    print("Initializing Vectorized Individual-Based Model (2000 cells)...")
    
    # Create Environment
    env = DummyVecEnv([lambda: GeneticPhotobioreactorEnv(max_cells=300000, initial_cells=500)])
    
    # Apply VecNormalize to handle large rewards and unscaled observations
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.0)
    
    model_name = "model_data/ppo_genetic_ibm"
    tensorboard_log = "./ppo_genetic_tensorboard/"
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=7200,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=tensorboard_log,
        device="auto" # Auto-detect (CUDA if available, else CPU)
    )
    
    print("Starting Training with Genetic Domain Randomization...")
    print("Agent will face a different 'Algal Strain' every episode.")
    
    model.learn(
        total_timesteps=300_000, # Increased for better convergence
        callback=GeneticMonitorCallback(),
        progress_bar=True
    )
    
    print("Training Complete.")
    model.save(model_name)
    env.save("model_data/vec_normalize.pkl") # Save normalization stats
    print(f"Model saved to {model_name}.zip")
    print("Normalization stats saved to model_data/vec_normalize.pkl")

    # --- Verification Run ---
    print("\n--- Verifying Robustness on 3 New Strains ---")
    obs = env.reset()
    for i in range(3):
        print(f"\nTest Run {i+1} (New Random Strain)...")
        # Get params from internal env
        # Note: input_env[0] is the way to access un-vec env
        current_params = env.envs[0].strain_params 
        print(f"Strain Params: {current_params}")
        
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            total_reward += reward[0]
            done = dones[0]
            
        print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
