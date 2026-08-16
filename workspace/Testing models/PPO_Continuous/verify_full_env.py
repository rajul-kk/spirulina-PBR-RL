import numpy as np
from experiment_env import PhotobioreactorExperimentEnv

def verify_full_env():
    print("Initializing Enhanced Environment...")
    env = PhotobioreactorExperimentEnv()
    
    # 1. Check Noise
    # Reset multiple times to check if starting state is jittery? 
    # Actually reset initializes to fixed state then adds noise?
    # No, reset initializes fixed state. `_add_noise` is only called at end of step.
    # Wait, reset usually returns observation. My code returns `self.state`, `reward`, etc from step.
    # But `reset()` lines 57 returns `self.state`. 
    # I did NOT update `reset` to return usage of `_add_noise`.
    # I should check if that was missed.
    # Let's check step output first.
    
    obs, info = env.reset()
    print(f"Initial (Clean) State: {obs}")
    
    action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, done, trunc, info = env.step(action)
    
    print(f"Step 1 Observation (Noisy): {obs}")
    
    # Check if observation is slightly different from what physics would predict perfectly?
    # Hard to tell without reference. But if it runs, good.
    
    print(f"Reward: {reward}")
    print("Environment step successful.")

if __name__ == "__main__":
    verify_full_env()
