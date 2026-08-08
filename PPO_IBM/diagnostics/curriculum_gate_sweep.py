"""
curriculum_gate_sweep.py — validates ADVANCE_TARGETS thresholds in curriculum_schedule.py
by running constant-action (best known stir/light/frac) physics probes at D0 and D1,
the two tiers whose thresholds were only ever scaled off a single D2 setpoint sweep
rather than measured directly.

Read-only physics probe — no model, no training. Same pattern as dynamic_profile_sweep.py,
but across difficulty tiers instead of harvest fractions, at the fraction
(0.15) that sweep found best-sustainable at D2.

Usage:
    python curriculum_gate_sweep.py
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

SEEDS = [0, 1, 2, 3, 4, 5]
STIR_RPM = 80.0
LIGHT_UMOL = 1000.0
FRAC = 0.15  # best-sustainable constant fraction found at D2 in dynamic_profile_sweep.py
INITIAL_CELLS = 300

GATES = {
    0: {"min_median_harvested_mg": 30.0, "min_p25_harvested_mg": 15.0, "max_crash_rate": 0.15, "min_median_time_avg_od": 0.004},
    1: {"min_median_harvested_mg": 60.0, "min_p25_harvested_mg": 30.0, "max_crash_rate": 0.10, "min_median_time_avg_od": 0.008},
    2: {"min_median_harvested_mg": 90.0, "min_p25_harvested_mg": 50.0, "max_crash_rate": 0.08, "min_median_time_avg_od": 0.011},
}


def to_raw(stir, light, frac, f_max):
    return np.array([
        np.interp(stir, [50, 200], [-1, 1]),
        np.interp(light, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def run_episode(seed, difficulty):
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=INITIAL_CELLS, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    action = to_raw(STIR_RPM, LIGHT_UMOL, FRAC, env.F_MAX)
    done = False
    step = 0
    info = {}
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    crashed = step < env.max_steps
    harvested_mg = float(info.get("cumulative_harvested_mg", 0.0))
    time_avg_od = float(info.get("time_avg_od", 0.0))
    return harvested_mg, time_avg_od, crashed, step


def main():
    print(f"Fixed stir={STIR_RPM}rpm light={LIGHT_UMOL}umol frac={FRAC} across D0/D1/D2, seeds={SEEDS}\n")
    for diff in [0, 1, 2]:
        harvests, ods, crashes = [], [], []
        for seed in SEEDS:
            h, t, c, steps = run_episode(seed, diff)
            harvests.append(h)
            ods.append(t)
            crashes.append(c)
            print(f"  D{diff} seed={seed}  harvested={h:7.1f}mg  time_avg_od={t:.4f}  crashed={c}  steps={steps}")
        harvests = np.array(harvests)
        ods = np.array(ods)
        crash_rate = float(np.mean(crashes))
        median_h = float(np.median(harvests))
        p25_h = float(np.percentile(harvests, 25))
        median_od = float(np.median(ods))

        gate = GATES[diff]
        print(f"\n  D{diff} summary: median_harvest={median_h:.1f} (gate>={gate['min_median_harvested_mg']})  "
              f"p25={p25_h:.1f} (gate>={gate['min_p25_harvested_mg']})  "
              f"crash={crash_rate*100:.0f}% (gate<={gate['max_crash_rate']*100:.0f}%)  "
              f"time_avg_od={median_od:.4f} (gate>={gate['min_median_time_avg_od']})")
        headroom_h = median_h / gate["min_median_harvested_mg"]
        headroom_od = median_od / gate["min_median_time_avg_od"] if gate["min_median_time_avg_od"] > 0 else float("inf")
        print(f"  headroom: harvest {headroom_h:.2f}x gate, time_avg_od {headroom_od:.2f}x gate\n")


if __name__ == "__main__":
    main()
