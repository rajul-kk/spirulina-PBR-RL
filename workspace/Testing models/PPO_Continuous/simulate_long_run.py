import numpy as np
from experiment_env import PhotobioreactorExperimentEnv
import matplotlib.pyplot as plt

def simulate_long_term():
    env = PhotobioreactorExperimentEnv()
    obs, info = env.reset()
    
    # Set max steps to 5000 for this test
    env.max_steps_per_episode = 5000
    
    print(f"Initial Density: {info['density']:.4f} g/L")
    
    densities = []
    times = []
    
    # Constant Max Light
    action = [0, 0, 0, 4] 
    
    for t in range(5000):
        obs, reward, term, trunc, info = env.step(action)
        densities.append(info['density'])
        times.append(t)
        
        if info['density'] >= 19.9: # Near saturation cap
            print(f"Saturated at t={t} hours")
            break
            
    print(f"Final Density: {densities[-1]:.4f} g/L after {len(densities)} hours")
    
    # ASCII Plot
    # Shows sigmoidal growth curve (as expected)
    print("\nLong-Term Growth Curve (sampled every 100h):")
    max_d = max(densities)
    for i in range(0, len(densities), 100):
        val = densities[i]
        bar = '#' * int(val / max_d * 50)
        print(f"{i:4d}h | {bar} {val:.2f} g/L")

if __name__ == "__main__":
    simulate_long_term()
