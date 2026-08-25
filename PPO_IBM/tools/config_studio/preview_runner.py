"""
preview_runner.py — the "live preview" half of config_studio. Runs the scripted
expert controller directly against genetic_env.py (no NN, same law used in
experiments/bc_scaffold/scripts/expert_sweep.py) using whatever values the UI has
staged (which may not be saved to disk yet), and scores the result against a gate
(either the file's current saved gate, or a staged one).

Deliberately independent of experiments/bc_scaffold/scripts/expert_sweep.py's
hardcoded constants — every parameter here is a function argument, because this is
exactly what needs to vary as the user drags sliders in the UI.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # PPO_IBM/
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from genetic_env import GeneticPhotobioreactorEnv


def sample_init_cells(rng, adversarial_frac=0.10):
    if rng.rand() < adversarial_frac:
        return int(rng.uniform(30, 80))
    return int(np.exp(rng.uniform(np.log(100), np.log(400))))


def run_episode(difficulty, init_cells, seed, stir, light, od_setpoint, gain, frac_cap):
    env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
    env.reset(seed=seed)
    f_max = float(getattr(env, "F_MAX", 0.5))

    done = False
    step = 0
    info = {}
    while not done:
        surplus = (float(getattr(env, "od", 0.0)) / od_setpoint) - 1.0
        frac = float(np.clip(gain * surplus, 0.0, frac_cap))
        action = np.array([
            np.interp(stir, [50, 200], [-1, 1]),
            np.interp(light, [0, 2000], [-1, 1]),
            np.interp(frac, [0, f_max], [-1, 1]),
        ], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    return {
        "crashed": step < env.max_steps,
        "harvested_mg": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
    }


def run_preview(difficulty, n_episodes, stir_min, stir_max, light_min, light_max,
                 od_setpoint, gain, frac_cap, gate, base_seed=1000):
    rng = np.random.RandomState(base_seed)
    ctrl_rng = np.random.RandomState(base_seed + 999)
    stir = float(ctrl_rng.uniform(stir_min, stir_max))
    light = float(ctrl_rng.uniform(light_min, light_max))

    episodes = []
    for i in range(n_episodes):
        seed = base_seed + i
        init_cells = sample_init_cells(rng)
        r = run_episode(difficulty, init_cells, seed, stir, light, od_setpoint, gain, frac_cap)
        r["seed"] = seed
        r["init_cells"] = init_cells
        episodes.append(r)

    harvested = np.array([e["harvested_mg"] for e in episodes])
    time_od = np.array([e["time_avg_od"] for e in episodes])
    crash_rate = float(np.mean([e["crashed"] for e in episodes]))
    med_h, p25_h = float(np.median(harvested)), float(np.percentile(harvested, 25))
    med_od = float(np.median(time_od))

    ok = (med_h >= gate["harvest"] and p25_h >= gate["p25"]
          and crash_rate <= gate["crash"] and med_od >= gate["od"])

    return {
        "stir": stir, "light": light,
        "episodes": episodes,
        "median_harvested_mg": med_h, "p25_harvested_mg": p25_h,
        "median_time_avg_od": med_od, "crash_rate": crash_rate,
        "gate": gate, "passes_gate": ok,
    }
