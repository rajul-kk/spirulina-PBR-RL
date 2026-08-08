"""
dynamic_profile_sweep_od.py — ad hoc extension of dynamic_profile_sweep.py that also reports
time_avg_od per frac, to check whether the D1 gate's two thresholds (median_harvested_mg>=60,
median_time_avg_od>=0.008) are jointly achievable under a fixed-action physics-only policy,
or whether they trade off against each other (as v15's deterministic-eval trace suggested).

Read-only physics probe — no model, no training.
"""

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
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environments"))
from genetic_env import GeneticPhotobioreactorEnv

SEEDS = [0, 1, 2, 3]
DIFFICULTY = 1
INITIAL_CELLS = 300

# Two operating points: the reference one used for the original sweep, and the one the v15
# trained policy actually converged to (measured via test_actions.py on both v15 archives:
# stir 55-63rpm — near the 50rpm floor — and light 875-930umol).
OPERATING_POINTS = [
    ("reference", 80.0, 1000.0),
    ("v15-policy", 60.0, 900.0),
]

FRAC_VALUES = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]


def to_raw(stir, light, frac, f_max):
    return np.array([
        np.interp(stir, [50, 200], [-1, 1]),
        np.interp(light, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def run_episode(seed, frac, stir_rpm, light_umol):
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=INITIAL_CELLS, difficulty=DIFFICULTY)
    obs, _ = env.reset(seed=seed)
    action = to_raw(stir_rpm, light_umol, frac, env.F_MAX)
    done = False
    step = 0
    total_reward = 0.0
    info = {}
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        step += 1
    crashed = step < env.max_steps
    harvested_mg = float(env.cumulative_harvested_mg)
    time_avg_od = float(info.get("time_avg_od", 0.0))
    return harvested_mg, time_avg_od, total_reward, crashed, step


def sweep(name, stir_rpm, light_umol):
    print(f"\n{'='*78}")
    print(f"  OPERATING POINT '{name}': D{DIFFICULTY}, stir={stir_rpm}rpm, light={light_umol}umol")
    print(f"{'='*78}")
    print(f"{'frac':>6} {'harvest_mg':>11} {'time_avg_od':>12} {'reward':>10} {'crash%':>8}  gate(h>=60,od>=0.008)")
    best_frac, best_reward = None, -1e18
    for frac in FRAC_VALUES:
        harvests, ods, rewards, crashes = [], [], [], []
        for seed in SEEDS:
            h, od, rew, crashed, steps = run_episode(seed, frac, stir_rpm, light_umol)
            harvests.append(h)
            ods.append(od)
            rewards.append(rew)
            crashes.append(crashed)
        mh, mo, mr, cr = np.median(harvests), np.median(ods), np.mean(rewards), np.mean(crashes)
        gate = "PASS" if (mh >= 60.0 and mo >= 0.008 and cr <= 0.10) else "fail"
        print(f"{frac:6.2f} {mh:11.1f} {mo:12.4f} {mr:10.1f} {cr*100:7.1f}%  {gate}")
        if mr > best_reward:
            best_frac, best_reward = frac, mr
    print(f"\n  --> REWARD-OPTIMAL frac at this operating point: {best_frac} (mean reward {best_reward:.1f})")


def main():
    for name, stir, light in OPERATING_POINTS:
        sweep(name, stir, light)


if __name__ == "__main__":
    main()
