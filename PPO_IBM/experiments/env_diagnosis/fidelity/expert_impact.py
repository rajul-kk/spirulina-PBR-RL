"""Scripted-expert yield on the current env vs another env file (e.g. the pre-audit version), over a
light x init grid at D2. Used to judge whether the fidelity fixes move the gates' scale.

  python experiments/env_diagnosis/fidelity/expert_impact.py [path/to/other_genetic_env.py]
"""
import importlib.util
import sys

import numpy as np

from common import MAX_CELLS, act
from genetic_env import GeneticPhotobioreactorEnv


def load(path):
    spec = importlib.util.spec_from_file_location("genetic_env_other", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GeneticPhotobioreactorEnv


def episode(cls, init, seed, light, stir=65.0, setpoint=0.6, cap=0.30, difficulty=2):
    np.random.seed(seed)
    env = cls(max_cells=MAX_CELLS, initial_cells=init, difficulty=difficulty)
    env.reset(seed=seed)
    done, t = False, 0
    while not done:
        frac = float(np.clip(env.od / setpoint - 1.0, 0.0, cap))
        _, _, term, trunc, info = env.step(act(stir, light, frac))
        done, t = term or trunc, t + 1
    return env.cumulative_harvested_mg, t < env.max_steps, info["time_avg_od"]


if __name__ == "__main__":
    cls = load(sys.argv[1]) if len(sys.argv) > 1 else GeneticPhotobioreactorEnv
    print(f"env: {sys.argv[1] if len(sys.argv) > 1 else 'current'}")
    for light in (1000.0, 1400.0, 2000.0):
        rows = [episode(cls, init, seed, light) for init in (60, 120, 250, 700, 2500) for seed in (101, 102)]
        h = np.array([r[0] for r in rows])
        print(f"  light {light:.0f}: harvest median {np.median(h):.0f} mg, p25 {np.percentile(h, 25):.0f}, "
              f"crashes {sum(r[1] for r in rows)}/{len(rows)}, time-avg od median {np.median([r[2] for r in rows]):.2f}")
        print("     per episode:", " ".join(f"{x:.0f}" for x in h))
        sys.stdout.flush()
