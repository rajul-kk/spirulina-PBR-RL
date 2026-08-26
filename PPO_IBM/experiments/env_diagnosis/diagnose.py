"""
diagnose.py — actions/reward/environment diagnostic sweep for GeneticPhotobioreactorEnv.

Investigates TD3+BC (v35)'s post-resume divergence (critic loss 7.8->543.6, crash rate
0%->100% over a few chunks; runs_registry.csv v35_td3bc) via three sweeps:

1. REWARD — per-step reward distribution under the scripted expert, and how much of an
   outlier the -100 crash/extinction penalty (genetic_env.py) is against it.
2. ENVIRONMENT — crash rate by initial-population bucket (low/mid/high) x difficulty,
   under the scripted expert and under random actions (proxy for a perturbed policy).
3. ACTIONS — harvest-fraction crash boundary ("washout cliff"), re-verified against the
   current env version.

Usage (from repo root, PPO_IBM/):
    python experiments/env_diagnosis/diagnose.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from genetic_env import GeneticPhotobioreactorEnv

EXPERT_STIR_RANGE = (60.0, 80.0)
EXPERT_LIGHT_RANGE = (900.0, 1000.0)
EXPERT_OD_SETPOINT = 0.015
EXPERT_GAIN = 1.0
EXPERT_FRAC_CAP = 0.30


def expert_harvest_frac(od):
    surplus = (float(od) / EXPERT_OD_SETPOINT) - 1.0
    return float(np.clip(EXPERT_GAIN * surplus, 0.0, EXPERT_FRAC_CAP))


def run_episode(difficulty, init_cells, seed, policy, fixed_frac=None, fixed_stir=None, fixed_light=None):
    """policy: 'expert' or 'random'. Returns (per-step rewards list, crashed, steps, term_sums)."""
    env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    stir = fixed_stir if fixed_stir is not None else float(rng.uniform(*EXPERT_STIR_RANGE))
    light = fixed_light if fixed_light is not None else float(rng.uniform(*EXPERT_LIGHT_RANGE))
    f_max = float(getattr(env, "F_MAX", 0.5))

    rewards = []
    done = False
    step = 0
    while not done:
        if policy == "random":
            action = env.action_space.sample()
        else:
            frac = fixed_frac if fixed_frac is not None else expert_harvest_frac(getattr(env, "od", 0.0))
            action = np.array([
                np.interp(stir, [50, 200], [-1, 1]),
                np.interp(light, [0, 2000], [-1, 1]),
                np.interp(frac, [0, f_max], [-1, 1]),
            ], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        done = terminated or truncated
        step += 1

    return rewards, step < env.max_steps, step, dict(env.reward_term_sums)


# ═══════════════════════════════════════════════════════════════════════════
# (full rationale: docs/decision_history.md#--experiments-env_diagnosis-diagnose-py-71)

def sweep_reward():
    print("\n" + "=" * 78)
    print("  SWEEP 1: REWARD MAGNITUDE AUDIT")
    print("=" * 78)

    all_rewards = []
    all_term_sums = []
    for seed in range(10):
        rewards, crashed, steps, term_sums = run_episode(2, 300, seed, "expert")
        all_rewards.extend(rewards)
        all_term_sums.append(term_sums)

    arr = np.array(all_rewards)
    print(f"  Scripted expert, D2, 10 episodes, {len(arr):,} per-step reward samples:")
    print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"    p1={np.percentile(arr,1):.4f}  p99={np.percentile(arr,99):.4f}")

    mean_term = {k: float(np.mean([t[k] for t in all_term_sums])) for k in all_term_sums[0]}
    print(f"  Per-episode term-sum means (7200-step episode): {mean_term}")

    CRASH_PENALTY = -100.0
    print(f"\n  Crash/extinction penalty: {CRASH_PENALTY}")
    print(f"    vs typical per-step reward (mean {arr.mean():.4f}): {abs(CRASH_PENALTY)/max(abs(arr.mean()),1e-9):.0f}x")
    print(f"    vs best single-step reward observed ({arr.max():.4f}): {abs(CRASH_PENALTY)/max(arr.max(),1e-9):.0f}x")
    print(f"    vs worst NON-crash single-step reward (p1={np.percentile(arr,1):.4f}): "
          f"{abs(CRASH_PENALTY)/max(abs(np.percentile(arr,1)),1e-9):.0f}x")

    gamma = 0.9995
    horizon = int(round(1 / (1 - gamma)))
    print(f"\n  GAMMA=0.9995 -> effective bootstrap horizon ~{horizon:,} steps.")
    print(f"  A single -100 crash penalty, discounted back {horizon:,} steps at this gamma, "
          f"is still visible in a Q-target the whole way (gamma^{horizon}={gamma**horizon:.3f} of "
          f"its value survives at the horizon edge) -- meaning a SEQ_LEN=60 truncated-BPTT "
          f"training window landing anywhere near a crash step bootstraps against a Q-target "
          f"dominated by that single outlier, not the dense per-step signal.")


# ═══════════════════════════════════════════════════════════════════════════
# (full rationale: docs/decision_history.md#--experiments-env_diagnosis-diagnose-py-112)

BUCKETS = {"low (100-400)": (100, 400), "mid (600-1500)": (600, 1500), "high (2000-5000)": (2000, 5000)}


def sweep_environment(n_per_cell=8):
    print("\n" + "=" * 78)
    print("  SWEEP 2: CRASH RATE BY INITIAL-POPULATION BUCKET x DIFFICULTY")
    print("=" * 78)
    print(f"  {n_per_cell} episodes per (bucket, difficulty, policy) cell\n")

    header = f"  {'bucket':<18}{'D':<3}{'expert crash%':<16}{'random crash%':<16}"
    print(header)
    for bucket_name, (lo, hi) in BUCKETS.items():
        for difficulty in (0, 1, 2):
            rng = np.random.default_rng(hash((bucket_name, difficulty)) % (2**31))
            expert_crashes, random_crashes = [], []
            for i in range(n_per_cell):
                init_cells = int(rng.uniform(lo, hi))
                seed = 5000 + i
                _, crashed_e, _, _ = run_episode(difficulty, init_cells, seed, "expert")
                _, crashed_r, _, _ = run_episode(difficulty, init_cells, seed + 500, "random")
                expert_crashes.append(crashed_e)
                random_crashes.append(crashed_r)
            print(f"  {bucket_name:<18}{difficulty:<3}"
                  f"{np.mean(expert_crashes)*100:<16.0f}{np.mean(random_crashes)*100:<16.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# (full rationale: docs/decision_history.md#--experiments-env_diagnosis-diagnose-py-142)

def sweep_actions(n_seeds=5):
    print("\n" + "=" * 78)
    print("  SWEEP 3: HARVEST-FRACTION CRASH BOUNDARY (fixed frac all episode, D2)")
    print("=" * 78)
    print(f"  {n_seeds} seeds per fraction, init_cells=300, stir=70, light=950\n")
    print(f"  {'frac':<8}{'crash%':<10}{'median_harvest_mg':<20}{'median_time_avg_od':<20}")

    for frac in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        crashes, harvests, ods = [], [], []
        for seed in range(n_seeds):
            env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=300, difficulty=2)
            env.reset(seed=8000 + seed)
            f_max = float(getattr(env, "F_MAX", 0.5))
            action = np.array([
                np.interp(70.0, [50, 200], [-1, 1]),
                np.interp(950.0, [0, 2000], [-1, 1]),
                np.interp(frac, [0, f_max], [-1, 1]),
            ], dtype=np.float32)
            done, step, info = False, 0, {}
            while not done:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1
            crashes.append(step < env.max_steps)
            harvests.append(float(info.get("cumulative_harvested_mg", 0.0)))
            ods.append(float(info.get("time_avg_od", 0.0)))
        print(f"  {frac:<8.2f}{np.mean(crashes)*100:<10.0f}{np.median(harvests):<20.1f}{np.median(ods):<20.4f}")


if __name__ == "__main__":
    sweep_reward()
    sweep_environment()
    sweep_actions()
    print("\nDone.")
