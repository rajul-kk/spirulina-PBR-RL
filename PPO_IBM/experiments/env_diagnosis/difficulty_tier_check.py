"""Do D0/D1/D2 actually differ in hardness, and does a fixed OD_TARGET mean the same
(full rationale: docs/decision_history.md#--experiments-env_diagnosis-difficulty_tier_check-py-1)"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from genetic_env import GeneticPhotobioreactorEnv

EXPERT_OD_SETPOINT = 0.015
EXPERT_FRAC_CAP = 0.30


def expert_action(env):
    surplus = (float(env.od) / EXPERT_OD_SETPOINT) - 1.0
    frac = float(np.clip(surplus, 0.0, EXPERT_FRAC_CAP))
    f_max = float(getattr(env, "F_MAX", 0.5))
    return np.array([
        np.interp(70.0, [50, 200], [-1, 1]),
        np.interp(950.0, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def episode(difficulty, init_cells, seed):
    env = GeneticPhotobioreactorEnv(max_cells=50_000, initial_cells=init_cells, difficulty=difficulty)
    env.reset(seed=seed)
    done, info = False, {}
    peak_od = 0.0
    while not done:
        obs, r, term, trunc, info = env.step(expert_action(env))
        peak_od = max(peak_od, float(env.od))
        done = term or trunc
    return {
        "harvest": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
        "peak_od": peak_od,
        "crashed": env.step_count < env.max_steps,
    }


def main():
    N = 6
    print("=" * 78)
    print("  A. DIFFICULTY TIERS — same expert, same cold starts, only `difficulty` varies")
    print("=" * 78)
    print(f"  {'tier':<6}{'harvest med':>13}{'time_avg_od':>14}{'peak_od':>10}{'crash%':>9}")
    for d in (0, 1, 2):
        res = [episode(d, 300, 100 + i) for i in range(N)]
        h = np.median([r["harvest"] for r in res])
        o = np.median([r["time_avg_od"] for r in res])
        pk = np.median([r["peak_od"] for r in res])
        c = 100 * np.mean([r["crashed"] for r in res])
        print(f"  D{d:<5}{h:>13.1f}{o:>14.4f}{pk:>10.4f}{c:>9.0f}")

    print("\n" + "=" * 78)
    print("  B. FIXED OD_TARGET (0.012) vs COLD-START SIZE — is the same target equally")
    print("     reachable from every bucket the curriculum samples?")
    print("=" * 78)
    print(f"  {'init_cells':<12}{'time_avg_od':>14}{'peak_od':>10}{'x OD_TARGET':>14}{'harvest':>10}")
    for init in (50, 150, 300, 900, 3000):
        res = [episode(2, init, 200 + i) for i in range(N)]
        o = np.median([r["time_avg_od"] for r in res])
        pk = np.median([r["peak_od"] for r in res])
        h = np.median([r["harvest"] for r in res])
        print(f"  {init:<12}{o:>14.4f}{pk:>10.4f}{pk / 0.012:>14.2f}{h:>10.1f}")


if __name__ == "__main__":
    main()
