"""
dynamic_profile_sweep.py — Part 4 diagnostic for the periodic semi-continuous harvest
redesign: sweep constant per-event harvest fractions (with the known best stir/light
combo held fixed) at 20L/D2 physics to find (a) the achievable per-event mg ceiling
(feeds TARGET_MG_PER_EVENT in genetic_env.py's _compute_reward) and (b) roughly where
repeated over-harvesting starts causing washout (sanity-checks F_MAX).

Read-only physics probe — no model, no training. Drives the env directly with raw
actions decoded the same way genetic_env.step() does (np.interp[-1,1] -> physical).

Usage:
    python dynamic_profile_sweep.py
"""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-dynamic_profile_sweep-py-15)
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
DIFFICULTY = 2
INITIAL_CELLS = 300
STIR_RPM = 80.0
LIGHT_UMOL = 1000.0  # best static combo from the earlier batch-mode grid sweep

# Harvest fractions to sweep, applied every harvest event (every HARVEST_INTERVAL_STEPS).
FRAC_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def to_raw(stir, light, frac, f_max):
    return np.array([
        np.interp(stir, [50, 200], [-1, 1]),
        np.interp(light, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def run_episode(seed, frac):
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=INITIAL_CELLS, difficulty=DIFFICULTY)
    obs, _ = env.reset(seed=seed)
    action = to_raw(STIR_RPM, LIGHT_UMOL, frac, env.F_MAX)
    max_steps = env.max_steps
    n_events_expected = max_steps // env.HARVEST_INTERVAL_STEPS
    total_reward = 0.0
    done = False
    step = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        step += 1
    crashed = step < max_steps
    harvested_mg = float(env.cumulative_harvested_mg)
    n_events_survived = step // env.HARVEST_INTERVAL_STEPS
    mg_per_event = harvested_mg / max(n_events_survived, 1)
    return total_reward, harvested_mg, mg_per_event, crashed, step, n_events_expected


def main():
    print(f"Fixed stir={STIR_RPM}rpm, light={LIGHT_UMOL}umol/m2/s, sweeping harvest_frac in {FRAC_VALUES}\n")
    results = {}
    for frac in FRAC_VALUES:
        rewards, harvests, per_event, crashes, steps_list = [], [], [], [], []
        for seed in SEEDS:
            r, h, mg_ev, crashed, steps, n_exp = run_episode(seed, frac)
            rewards.append(r)
            harvests.append(h)
            per_event.append(mg_ev)
            crashes.append(crashed)
            steps_list.append(steps)
        results[frac] = {
            "mean_reward": float(np.mean(rewards)),
            "mean_harvested_mg": float(np.mean(harvests)),
            "mean_mg_per_event": float(np.mean(per_event)),
            "crash_rate": float(np.mean(crashes)),
        }
        print(f"frac={frac:4.2f}  mean_harvested={np.mean(harvests):8.1f}mg  "
              f"mean_mg/event={np.mean(per_event):7.2f}mg  mean_reward={np.mean(rewards):7.1f}  "
              f"crash_rate={np.mean(crashes)*100:5.1f}%  steps={steps_list}")

    # Best sustainable (0% crash) fraction by mean harvested mg
    sustainable = {f: r for f, r in results.items() if r["crash_rate"] == 0.0}
    if sustainable:
        best_f = max(sustainable, key=lambda f: sustainable[f]["mean_harvested_mg"])
        print(f"\n{'='*90}")
        print(f"  BEST SUSTAINABLE frac (0% crash): {best_f} -> {results[best_f]['mean_harvested_mg']:.1f}mg total, "
              f"{results[best_f]['mean_mg_per_event']:.2f}mg/event")
        print(f"{'='*90}\n")
    else:
        print("\nNo fully sustainable (0% crash) fraction found in sweep range.\n")


if __name__ == "__main__":
    main()
