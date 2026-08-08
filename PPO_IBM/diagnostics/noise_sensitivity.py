"""
noise_sensitivity.py — how much does ACTION NOISE destroy each policy?

WHY: reward_ab.py showed the reward function ranks the scripted expert +313 above v17
(1079 vs 766), driven entirely by reward_od. So the reward is NOT exploitable and the
reward-structure hypothesis is dead. Yet PPO drifted from expert-like behaviour to v17's.

PPO maximises expected reward UNDER ITS OWN SAMPLING NOISE; we evaluate deterministically.
If the expert's strategy is noise-fragile (it harvests ~0 early, and Gaussian noise around
0 forces harvesting anyway because the low side clips) while v17's higher-baseline strategy
is noise-robust, then PPO was correctly optimising a DIFFERENT objective than the one the
curriculum gates score. That is the "stochastic-train / deterministic-evaluate" gap
documented in arXiv 2509.19464, which notes it widens on long-horizon tasks (ours: 7200 steps).

This script measures that directly: run each policy with Gaussian noise of varying sigma
added to its raw [-1,1] action, and see where each one's reward and time_avg_od fall apart.

Read-only. No training, no model modification.

Usage:
    python noise_sensitivity.py --model <path> --norm <path> --n 4
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
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environments"))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from genetic_env import GeneticPhotobioreactorEnv

EXPERT_OD_SETPOINT = 0.015
EXPERT_GAIN = 1.0
EXPERT_FRAC_CAP = 0.30
EXPERT_STIR = 70.0
EXPERT_LIGHT = 950.0

# train/std observed in v15/v16b/v17 hovered at ~0.50-0.53 and never fell below ~0.49.
SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]


def make_vec(norm_path, difficulty, init_cells):
    def _make():
        return Monitor(GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells,
                                                 difficulty=difficulty))
    vec = VecNormalize.load(norm_path, venv=DummyVecEnv([_make]))
    vec.training = False
    vec.norm_reward = False
    return vec


def raw_env_of(vec):
    e = vec.venv if hasattr(vec, "venv") else vec
    e = e.envs[0]
    return e.env if hasattr(e, "env") else e


def encode(stir, light, frac, f_max):
    return np.array([
        np.interp(stir, [50, 200], [-1, 1]),
        np.interp(light, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def run(vec, policy_fn, seed, sigma, rng):
    np.random.seed(seed)
    obs = vec.reset()
    raw = raw_env_of(vec)
    state = {"lstm": None, "starts": np.ones((1,), dtype=bool)}
    done, info, total = False, {}, 0.0
    f_max = float(getattr(raw, "F_MAX", 0.5))
    while not done:
        action = policy_fn(obs, raw, state, f_max)
        if sigma > 0.0:
            action = np.clip(action + rng.normal(0.0, sigma, size=action.shape), -1.0, 1.0
                             ).astype(np.float32)
        obs, reward, done_vec, info_list = vec.step(action)
        total += float(reward[0])
        done = bool(done_vec[0])
        info = info_list[0] if info_list else {}
    return total, float(info.get("cumulative_harvested_mg", 0.0)), float(info.get("time_avg_od", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=4000)
    args = ap.parse_args()

    init_rng = np.random.RandomState(args.base_seed)
    inits = [int(np.exp(init_rng.uniform(np.log(100), np.log(400)))) for _ in range(args.n)]

    probe = make_vec(args.norm, args.difficulty, inits[0])
    model = RecurrentPPO.load(args.model, env=probe, device="cpu",
                              custom_objects={"observation_space": probe.observation_space})

    def model_policy(obs, raw, state, f_max):
        act, state["lstm"] = model.predict(obs, state=state["lstm"],
                                           episode_start=state["starts"], deterministic=True)
        state["starts"] = np.zeros((1,), dtype=bool)
        return act

    def expert_policy(obs, raw, state, f_max):
        surplus = (float(getattr(raw, "od", 0.0)) / EXPERT_OD_SETPOINT) - 1.0
        frac = float(np.clip(EXPERT_GAIN * surplus, 0.0, EXPERT_FRAC_CAP))
        return encode(EXPERT_STIR, EXPERT_LIGHT, frac, f_max).reshape(1, -1)

    print(f"  Action-noise sensitivity, D{args.difficulty}, {args.n} episodes per sigma")
    print(f"  (v15/v16b/v17 all trained with train/std ~0.50)\n")
    print(f"  {'sigma':>6} | {'EXPERT reward':>14} {'harv':>7} {'od':>8} "
          f"| {'v17 reward':>11} {'harv':>7} {'od':>8}")
    print(f"  {'-'*6} | {'-'*14} {'-'*7} {'-'*8} | {'-'*11} {'-'*7} {'-'*8}")

    results = {}
    for sigma in SIGMAS:
        row = {}
        for name, fn in (("expert", expert_policy), ("v17", model_policy)):
            rs, hs, ods = [], [], []
            for i, init in enumerate(inits):
                rng = np.random.RandomState(args.base_seed + 7919 * i)
                vec = make_vec(args.norm, args.difficulty, init)
                r, h, od = run(vec, fn, args.base_seed + i, sigma, rng)
                vec.close()
                rs.append(r); hs.append(h); ods.append(od)
            row[name] = (float(np.mean(rs)), float(np.mean(hs)), float(np.mean(ods)))
        results[sigma] = row
        e, v = row["expert"], row["v17"]
        print(f"  {sigma:>6.2f} | {e[0]:>14.1f} {e[1]:>7.1f} {e[2]:>8.4f} "
              f"| {v[0]:>11.1f} {v[1]:>7.1f} {v[2]:>8.4f}")

    print()
    e0 = results[0.0]["expert"][0]
    crossover = None
    for sigma in SIGMAS:
        if results[sigma]["v17"][0] >= results[sigma]["expert"][0] and crossover is None:
            crossover = sigma
    print(f"  Expert reward at sigma=0: {e0:.1f}")
    for sigma in SIGMAS:
        e = results[sigma]["expert"]
        print(f"    sigma={sigma:.2f}: expert keeps {100.0*e[0]/max(e0,1e-9):5.1f}% of its "
              f"noise-free reward, time_avg_od {e[2]:.4f}")
    if crossover is not None:
        print(f"\n  CROSSOVER: at sigma>={crossover:.2f} the trained policy scores >= the expert.")
        print(f"  If that sigma is at or below the ~0.50 train/std these runs actually used,")
        print(f"  PPO was correctly optimising its noisy objective, and the fix is to reduce")
        print(f"  action noise (entropy), NOT to change the reward.")
    else:
        print("\n  No crossover in the tested range: the expert dominates at every sigma,")
        print("  which would mean noise alone does NOT explain the drift.")


if __name__ == "__main__":
    main()
