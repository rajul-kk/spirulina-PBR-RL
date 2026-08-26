
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-_verify_envs-py-2)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import sys
sys.path.insert(0, 'environments')
from genetic_env import GeneticPhotobioreactorEnv
import numpy as np

action = np.array([-0.52, -0.40, 0.0, -1.0], dtype=np.float32)

for diff in [0, 1, 2]:
    env = GeneticPhotobioreactorEnv(initial_cells=5000, difficulty=diff)
    env.reset()
    ep_reward = 0.0
    for step in range(2400):
        obs, r, done, _, _ = env.step(action)
        ep_reward += r
        if done:
            break
    status = "OK" if env.num_active > 4000 else "WARN (pop dropped)"
    print(f"D{diff}: pop={env.num_active}  pH={env.ph:.2f}  DO2={env.do2:.1f}  R={ep_reward:.1f}  [{status}]")
    env.close()
