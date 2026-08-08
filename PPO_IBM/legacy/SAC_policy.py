
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'environments'))
from genetic_env import GeneticPhotobioreactorEnv

def train_sac_agent():
    print("--- SAC Agent (High Efficiency / Entropy Maximization) ---")
    print("Initializing Environment...")
    
    # 1. Create base env
    # Match the configuration of PPO/RecurrentPPO (max_cells=300000)
    env = DummyVecEnv([lambda: GeneticPhotobioreactorEnv(max_cells=300000, initial_cells=500)])
    
    # 2. Wrap with VecNormalize
    # Critical for this env to handle large reward scaling
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.0)

    tensorboard_log = "./sac_tensorboard/"
    model_name = "model_data/sac_genetic_ibm"

    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,   # The Replay Buffer size
        learning_starts=1000,  # Collect some data before starting to learn
        batch_size=256,        # Larger batches are typical for SAC
        tau=0.005,             # Soft update coefficient
        gamma=0.99,            # Discount factor
        ent_coef="auto",       # SAC will automatically tune exploration (Entropy)
        train_freq=1,          # Update the model every step
        gradient_steps=1,      # Number of gradient steps per update
        tensorboard_log=tensorboard_log,
        device="auto"          # Auto-detect GPU
    )

    # Checkpoint Callback: Save every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./model_data/sac_checkpoints/',
        name_prefix='sac_ibm'
    )

    print("Training SAC Agent...")
    # SAC usually needs fewer timesteps than PPO to reach the same performance
    model.learn(total_timesteps=150000, progress_bar=True, log_interval=10, callback=checkpoint_callback)
    
    print("Training Complete.")
    model.save(model_name)
    env.save("model_data/sac_vec_normalize.pkl")
    print(f"Model saved to {model_name}.zip")
    print("Normalization stats saved to model_data/sac_vec_normalize.pkl")
    return model

if __name__ == "__main__":
    train_sac_agent()
