
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-test_why_die-py-2)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'environments'))
from environments.genetic_env import GeneticPhotobioreactorEnv

def test_why_die():
    env = GeneticPhotobioreactorEnv(initial_cells=3000, difficulty=0)
    obs, _ = env.reset()
    
    action = np.array([-0.0, 1.0, 1.0, 0.5])  
    with open('why_die_log.txt', 'w', encoding='utf-8') as f:
        for step in range(3000):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if env.num_active > 5800 and step % 10 == 0:
                f.write(f"--- Step {step} --- Pop: {env.num_active}\n")
                f.write(f"pH: {env.ph:.2f}, CO2: {env.dissolved_co2:.4f}, Nutrients: {env.ext_nutrients:.2f}\n")
                f.write(f"f_I: {env.debug_f_I:.4f}, f_pH: {env.debug_f_pH:.4f}, mu: {env.debug_mu:.4f}, stress: {env.debug_stress:.4f}\n")
            
if __name__ == "__main__":
    test_why_die()
