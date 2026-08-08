
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = r'E:\SEGP\PPO_IBM\ppo_recurrent_tensorboard\recurrent_ppo_curriculum_0'
ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
ea.Reload()

rew = ea.Scalars('rollout/ep_rew_mean')
ev  = ea.Scalars('train/explained_variance')
std = ea.Scalars('train/std')
pgl = ea.Scalars('train/policy_gradient_loss')
vl  = ea.Scalars('train/value_loss')
ent = ea.Scalars('train/entropy_loss')

targets = [500000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 7200000]
print("Step              rew_mean   exp_var   std     pg_loss    val_loss   entropy")
for t in targets:
    r = min(rew, key=lambda e: abs(e.step - t))
    e = min(ev,  key=lambda e: abs(e.step - t))
    s = min(std, key=lambda e: abs(e.step - t))
    p = min(pgl, key=lambda e: abs(e.step - t))
    v = min(vl,  key=lambda e: abs(e.step - t))
    n = min(ent, key=lambda e: abs(e.step - t))
    print(f"{r.step:>12,}   {r.value:>8.3f}   {e.value:>6.4f}   {s.value:>5.4f}   {p.value:>9.6f}   {v.value:>8.5f}   {n.value:>7.4f}")
