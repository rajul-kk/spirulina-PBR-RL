
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-test_heavy-py-2)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'environments'))
from heavy_env import HeavyPhotobioreactorEnv
import numpy as np

def test_heavy_env():
    print("Testing HeavyPhotobioreactorEnv...")
    env = HeavyPhotobioreactorEnv(max_cells=3000, initial_cells=500)
    obs, info = env.reset()
    
    # Run for a few steps to trigger division
    for i in range(100):
        # Use high light/nutrient to force rapid growth
        action = np.array([0.5, 1.0, 1.0, 0.5]) 
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            print(f"Episode ended at step {i}")
            break
        if i % 20 == 0:
            print(f"Step {i}: Total Cells = {info['pop']}")
            
    print("Test completed successfully!")

if __name__ == "__main__":
    test_heavy_env()
