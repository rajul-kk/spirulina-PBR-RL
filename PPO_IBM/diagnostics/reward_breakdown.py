"""
reward_breakdown.py — reports per-term reward contribution (od / biomass / stagnation /
washout / harvest) for a trained checkpoint, summed across full episodes. Diagnoses
whether any _compute_reward term in genetic_env.py is dead weight or dominating.

Requires genetic_env.py's reward_term_sums tracking (added alongside this script).

Usage:
    python reward_breakdown.py --model model_data/archive_.../recurrent_ppo_genetic_ibm --n 8
"""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-reward_breakdown-py-12)
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


def sample_init_cells(rng):
    if rng.rand() < 0.10:
        return int(rng.uniform(30, 80))
    return int(np.exp(rng.uniform(np.log(100), np.log(400))))


def run_episode(model, norm_path, difficulty, init_cells, seed):
    def _make():
        env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
        return Monitor(env)
    base = DummyVecEnv([_make])
    vec_env = VecNormalize.load(norm_path, venv=base)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()
    lstm_states = None
    ep_starts = np.ones((1,), dtype=bool)
    done = False
    terms = {}
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
        ep_starts = np.zeros((1,), dtype=bool)
        obs, reward, done_vec, info = vec_env.step(action)
        done = bool(done_vec[0])
        if done:
            terms = info[0].get("reward_term_sums", {})
    return terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--norm", type=str, default="model_data/recurrent_vec_normalize.pkl")
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=2000)
    args = ap.parse_args()

    print(f"Loading model {args.model} ...")
    model = RecurrentPPO.load(args.model)

    rng = np.random.RandomState(args.base_seed)
    all_terms = []
    for i in range(args.n):
        seed = args.base_seed + i
        init_cells = sample_init_cells(rng)
        np.random.seed(seed)
        terms = run_episode(model, args.norm, args.difficulty, init_cells, seed)
        all_terms.append(terms)
        total = sum(terms.values())
        parts = "  ".join(f"{k}={v:+7.1f}" for k, v in terms.items())
        print(f"  [{i+1}/{args.n}] seed={seed} init={init_cells:5d}  {parts}  total={total:+7.1f}")

    keys = ["od", "biomass", "od_delta", "harvest"]
    print(f"\n{'='*70}")
    print(f"  REWARD TERM BREAKDOWN  (mean across {args.n} episodes, D{args.difficulty})")
    print(f"{'='*70}")
    grand_total = 0.0
    means = {}
    for k in keys:
        vals = np.array([t.get(k, 0.0) for t in all_terms])
        means[k] = float(np.mean(vals))
        grand_total += means[k]
    for k in keys:
        share = (means[k] / grand_total * 100) if grand_total != 0 else 0.0
        print(f"  {k:12s}: mean={means[k]:+8.2f}   share_of_total={share:6.1f}%")
    print(f"  {'TOTAL':12s}: mean={grand_total:+8.2f}")


if __name__ == "__main__":
    main()
