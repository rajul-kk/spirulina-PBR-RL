
import numpy as np
from genetic_env import GeneticPhotobioreactorEnv

def test_static_hold(pop_size):
    print(f"--- Testing Plateau at Pop: {pop_size} (Smart Light) ---")
    env = GeneticPhotobioreactorEnv(max_cells=500_000, initial_cells=pop_size) # Increased max cap
    env.max_steps = 1000 
    obs, _ = env.reset()
    
    pop_history = []
    
    try:
        for t in range(200):
            # Smart Action: Light=500uE (-0.5), Nutrients=Max, CO2=Max, Stir=Max
            action = np.array([1.0, -0.5, 1.0, 1.0]) 
            obs, reward, done, _, info = env.step(action)
            pop_history.append(env.num_active)
    except Exception as e:
        print(e)
        
    delta = pop_history[-1] - pop_history[0]
    print(f"Start: {pop_history[0]}, End: {pop_history[-1]}, Delta: {delta}")

test_static_hold(150000) 
test_static_hold(180000) 
test_static_hold(220000)
