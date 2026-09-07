"""Measure reward-term scale: is the crash penalty a proportionate signal or an outlier?
(full rationale: docs/decision_history.md#--experiments-env_diagnosis-reward_scale_check-py-1)"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from genetic_env import GeneticPhotobioreactorEnv

EXPERT_STIR = 70.0
EXPERT_LIGHT = 950.0
EXPERT_OD_SETPOINT = 0.015
EXPERT_GAIN = 1.0
EXPERT_FRAC_CAP = 0.30


def expert_action(env):
    surplus = (float(env.od) / EXPERT_OD_SETPOINT) - 1.0
    frac = float(np.clip(EXPERT_GAIN * surplus, 0.0, EXPERT_FRAC_CAP))
    f_max = float(getattr(env, "F_MAX", 0.5))
    return np.array([
        np.interp(EXPERT_STIR, [50, 200], [-1, 1]),
        np.interp(EXPERT_LIGHT, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def run(difficulty, init_cells, seed, policy="expert", steps=None):
    env = GeneticPhotobioreactorEnv(max_cells=50_000, initial_cells=init_cells, difficulty=difficulty)
    env.reset(seed=seed)
    rewards, ods, pops = [], [], []
    done = False
    n = 0
    while not done and (steps is None or n < steps):
        a = expert_action(env) if policy == "expert" else env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        rewards.append(float(r))
        ods.append(float(env.od))
        pops.append(int(env.num_active))
        done = term or trunc
        n += 1
    crashed = env.step_count < env.max_steps
    return np.array(rewards), np.array(ods), np.array(pops), crashed


def main():
    print("=" * 74)
    print("  REWARD SCALE AUDIT")
    print("=" * 74)

    all_r = []
    for seed in range(8):
        r, _, _, _ = run(2, 300, seed, "expert")
        all_r.append(r)
    r = np.concatenate(all_r)
    mean_r, p1, p99 = r.mean(), np.percentile(r, 1), np.percentile(r, 99)
    print(f"  expert per-step reward (D2, 8 eps, n={len(r):,}):")
    print(f"    mean={mean_r:.4f}  std={r.std():.4f}  p1={p1:.4f}  p99={p99:.4f}  max={r.max():.4f}")

    probe = GeneticPhotobioreactorEnv(max_cells=1000, initial_cells=100, difficulty=0)
    pen = float(getattr(probe, "CRASH_PENALTY", 100.0))
    print(f"\n  crash penalty in use: -{pen}")
    print(f"    vs mean per-step reward : {pen / max(abs(mean_r), 1e-9):.0f}x")
    print(f"    vs worst non-crash step : {pen / max(abs(p1), 1e-9):.0f}x")
    print(f"    vs a full 7200-step episode ({mean_r * 7200:.0f} total): "
          f"{100 * pen / max(abs(mean_r * 7200), 1e-9):.1f}% of an episode")

    # graduated warning: how much signal exists as a culture approaches extinction?
    print("\n  near-extinction warning signal (random policy, low starts, D2):")
    found = 0
    for seed in range(40):
        if found >= 3:
            break
        rr, oo, pp, crashed = run(2, 60, 5000 + seed, "random")
        if crashed and len(rr) > 60:
            tail = rr[-50:]
            pre = rr[-50:-1]
            print(f"    crash@step{len(rr)}: mean reward last-50 (excl. terminal)={pre.mean():+.4f}, "
                  f"terminal step={rr[-1]:+.2f}, pop last-50 {pp[-50]}->{pp[-1]}")
            found += 1
    if found == 0:
        print("    (no crashes observed in sample)")


if __name__ == "__main__":
    main()
