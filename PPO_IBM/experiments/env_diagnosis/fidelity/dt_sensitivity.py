"""Time-step sensitivity: rerun the same physical schedule at dt = 0.02 h and dt = 0.01 h (step
counts, harvest interval and episode length scaled to keep the same hours) and compare the
trajectories. A consistent model converges; per-step constants show up as a systematic gap.

  python experiments/env_diagnosis/fidelity/dt_sensitivity.py [hours]
"""
import sys

import numpy as np

from common import MAX_CELLS, act
from genetic_env import GeneticPhotobioreactorEnv


def make_env(dt, init_cells, difficulty, seed):
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=difficulty)
    k = int(round(0.02 / dt))
    env.dt = dt
    env.HARVEST_INTERVAL_STEPS = 600 * k
    env.max_steps = 7200 * k
    env.BACK_HALF_STEP = 3600 * k
    env.reset(seed=seed)
    return env, k


def trajectory(dt, hours, stir, light, harvest_setpoint, seed, init_cells=700, difficulty=0):
    env, k = make_env(dt, init_cells, difficulty, seed)
    out = []
    for t in range(int(round(hours / dt))):
        frac = 0.0 if harvest_setpoint is None else float(np.clip(env.od / harvest_setpoint - 1.0, 0.0, 0.3))
        env.step(act(stir, light, frac))
        if (t + 1) % (50 * k) == 0:   # every hour
            am = env.active_mask
            out.append((env.od, float(np.mean(env.clump_mass[am])), env.membrane_integrity, env.temp,
                        env.cumulative_harvested_mg))
    return np.array(out)


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 72.0
    cases = [("50 rpm 1400 umol, no harvest", 50, 1400, None),
             ("65 rpm 1400 umol, expert harvest", 65, 1400, 0.6),
             ("150 rpm 1400 umol, no harvest", 150, 1400, None)]
    seeds = (1, 2, 3)
    for name, stir, light, sp in cases:
        res = {}
        for dt in (0.02, 0.01):
            res[dt] = np.mean([trajectory(dt, hours, stir, light, sp, s) for s in seeds], axis=0)
        a, b = res[0.02], res[0.01]
        print(f"\n=== {name}, {hours:.0f} h, mean of {len(seeds)} seeds (D0)")
        print("   hour |   od dt.02  od dt.01 | clump .02  clump .01 | membr .02 membr .01 | harvest .02 harvest .01")
        for h in list(range(11, len(a), 12)) + ([len(a) - 1] if (len(a) - 1) % 12 != 11 else []):
            print(f"   {h + 1:4d} | {a[h, 0]:9.3f} {b[h, 0]:9.3f} | {a[h, 1]:9.2f} {b[h, 1]:9.2f} | "
                  f"{a[h, 2]:8.3f} {b[h, 2]:8.3f} | {a[h, 4]:10.0f} {b[h, 4]:10.0f}")
        sys.stdout.flush()
