"""
held_out_sweep_harvest_fixed.py — independent held-out sweep for the v37 harvest-fixed
PPO ablation (experiments/harvest_ablation/). Same methodology as diagnostics/held_out_sweep.py
(90% lognormal(100,400) / 10% adversarial 30-80 cold starts, deterministic policy), but wraps
the env with HarvestFixedWrapper so harvest is fixed at frac=0.15 exactly as during training.

Usage (from repo root, PPO_IBM/):
    python experiments/harvest_ablation/held_out_sweep_harvest_fixed.py --n 40 --difficulty 1
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from genetic_env import GeneticPhotobioreactorEnv
from wrappers import HarvestFixedWrapper

FIXED_HARVEST_RAW = -0.4  # frac=0.15


def sample_init_cells(seed_rng, adversarial_frac=0.10):
    if seed_rng.rand() < adversarial_frac:
        return int(seed_rng.uniform(30, 80))
    return int(np.exp(seed_rng.uniform(np.log(100), np.log(400))))


def run_episode(model, norm_path, difficulty, init_cells, seed):
    def _make():
        env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
        env = HarvestFixedWrapper(env, FIXED_HARVEST_RAW)
        return Monitor(env)
    base = DummyVecEnv([_make])
    vec_env = VecNormalize.load(norm_path, venv=base)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()
    max_steps = vec_env.venv.envs[0].env.unwrapped.max_steps
    lstm_states = None
    ep_starts = np.ones((1,), dtype=bool)
    done = False
    step = 0
    harvested_mg = 0.0
    time_avg_od = 0.0
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
        ep_starts = np.zeros((1,), dtype=bool)
        obs, reward, done_vec, info = vec_env.step(action)
        done = bool(done_vec[0])
        step += 1
        if done:
            harvested_mg = float(info[0].get("cumulative_harvested_mg", 0.0))
            time_avg_od = float(info[0].get("time_avg_od", 0.0))

    crashed = step < max_steps
    return {"seed": seed, "init_cells": init_cells, "steps": step, "crashed": crashed,
            "harvested_mg": harvested_mg, "time_avg_od": time_avg_od}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="model_data/harvest_fixed_ppo/recurrent_ppo_genetic_ibm")
    ap.add_argument("--norm", type=str, default="model_data/harvest_fixed_ppo/vec_normalize.pkl")
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    print(f"Loading model {args.model} ...")
    model = RecurrentPPO.load(args.model)

    rng = np.random.RandomState(args.base_seed)
    results = []
    n_adversarial = 0
    for i in range(args.n):
        seed = args.base_seed + i
        init_cells = sample_init_cells(rng)
        if init_cells <= 80:
            n_adversarial += 1
        np.random.seed(seed)
        r = run_episode(model, args.norm, args.difficulty, init_cells, seed)
        results.append(r)
        tag = "ADV" if init_cells <= 80 else "   "
        print(f"  [{i+1:3d}/{args.n}] seed={seed:5d} {tag} init={init_cells:5d}  "
              f"steps={r['steps']:5d}  crashed={r['crashed']!s:5}  "
              f"harvested={r['harvested_mg']:7.1f}mg  time_avg_od={r['time_avg_od']:.4f}")

    crashes = [r["crashed"] for r in results]
    harvested = np.array([r["harvested_mg"] for r in results])
    time_od = np.array([r["time_avg_od"] for r in results])
    crash_rate = float(np.mean(crashes))

    print(f"\n{'='*70}")
    print(f"  HELD-OUT SWEEP — harvest-fixed PPO (D{args.difficulty}, n={args.n}, {n_adversarial} adversarial cold starts)")
    print(f"{'='*70}")
    print(f"  crash_rate           : {crash_rate*100:.1f}%")
    cvar_cutoff = np.percentile(harvested, 10)
    tail = harvested[harvested <= cvar_cutoff]
    cvar10 = float(np.mean(tail)) if len(tail) > 0 else float(harvested.min())
    print(f"  harvested_mg  median : {np.median(harvested):.1f}   p25: {np.percentile(harvested,25):.1f}   "
          f"cvar10: {cvar10:.1f}   min: {harvested.min():.1f}   max: {harvested.max():.1f}")
    print(f"  time_avg_od   median : {np.median(time_od):.4f}   p25: {np.percentile(time_od,25):.4f}")

    adv_results = [r for r in results if r["init_cells"] <= 80]
    if adv_results:
        adv_crash = np.mean([r["crashed"] for r in adv_results])
        print(f"  adversarial-only crash_rate ({len(adv_results)} eps): {adv_crash*100:.1f}%")


if __name__ == "__main__":
    main()
