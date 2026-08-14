"""
tdmpc2_held_out_sweep.py — read-only held-out robustness check for a trained TD-MPC2 checkpoint.

TD-MPC2 equivalent of held_out_sweep.py: that script assumes an SB3 model.predict() interface
(TDMPC2Agent.plan() is not compatible), so this is a separate, parallel implementation rather
than a shared one. Same project rule applies regardless: no mastery claim is final without an
independent held-out check on FRESH seeds, disjoint from anything used to gate training.

Both eval modes sample initial_cells via curriculum_schedule._sample_init_cells, matched to
difficulty (the run_tdmpc2_eval_episode / v27-diagnostic fix — NOT the training run's own
det-eval harness, which hardcoded initial_cells=3000 and is why this script exists as an
independent check rather than trusting the training loop's own [Det] numbers).

Usage:
    python diagnostics/tdmpc2_held_out_sweep.py --difficulty 0 --seeds 40
    python diagnostics/tdmpc2_held_out_sweep.py --difficulty 1 --seeds 40 --stochastic
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "legacy"), os.path.join(_ROOT, "training"),
           os.path.join(_ROOT, "environments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from TD_MPC2 import TDMPC2Agent, OBS_DIM, ACTION_DIM, MACRO_STEPS, ObservationBuffer
from curriculum_schedule import ADVANCE_TARGETS, _sample_init_cells, _compute_curriculum_stats


def run_episode(agent, difficulty, seed, stochastic=False, noise_scale=0.01,
                 horizon=12, num_samples=64):
    from genetic_env import GeneticPhotobioreactorEnv
    rng = np.random.default_rng(seed)
    init_cells = _sample_init_cells("random", difficulty)

    env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
    obs_buf = ObservationBuffer(obs_dim=OBS_DIM, order=16)
    raw_obs, _ = env.reset(seed=seed)
    obs_buf.reset(raw_obs, device=agent.device)
    obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
    obs_buf.set_state(m_t)

    done, step, info = False, 0, {}
    action = np.zeros(agent.action_dim, dtype=np.float32)
    while not done:
        if step % MACRO_STEPS == 0:
            action = agent.plan(raw_obs, obs_buf.get_state(), horizon=horizon,
                               num_samples=num_samples, num_iters=3)
            if stochastic:
                action = action + rng.normal(0, noise_scale, size=ACTION_DIM)
                action = np.clip(action, -1.0, 1.0)
        raw_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
        obs_buf.set_state(m_t)
        step += 1

    max_steps = env.max_steps
    return {
        "harvested_mg": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
        "crashed": step < max_steps,
        "start_mode": "low",
        "train_diff": difficulty,
        "reward": 0.0,
        "init_cells": init_cells,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="model_data/tdmpc2_genetic_ibm.pth")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions with the training-time exploration noise floor "
                         "(0.01 std, matching global_step > 30%% of budget) instead of the "
                         "bare plan() output.")
    ap.add_argument("--seed-offset", type=int, default=100_000,
                    help="Seeds start here — disjoint from any seed range used during "
                         "training's own det-eval calls.")
    args = ap.parse_args()

    print(f"  Model      : {args.model}")
    print(f"  Difficulty : D{args.difficulty}")
    print(f"  Mode       : {'stochastic (noise=0.01)' if args.stochastic else 'deterministic'}")
    print(f"  Seeds      : {args.seeds} (offset {args.seed_offset}, disjoint from training)")

    agent = TDMPC2Agent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cpu")
    agent.load(args.model)
    print(f"  Loaded.\n")

    records = []
    for i in range(args.seeds):
        seed = args.seed_offset + i
        rec = run_episode(agent, args.difficulty, seed, stochastic=args.stochastic)
        records.append(rec)
        print(f"  seed={seed:>7}  init_cells={rec['init_cells']:>5}  "
              f"harvest_mg={rec['harvested_mg']:>7.1f}  time_avg_od={rec['time_avg_od']:.4f}  "
              f"crashed={rec['crashed']}")

    stats = _compute_curriculum_stats(records, mastery_diff=args.difficulty)
    target = ADVANCE_TARGETS.get(args.difficulty, {})

    print(f"\n{'='*70}")
    print(f"  HELD-OUT SWEEP RESULT — D{args.difficulty}, n={stats['episodes']}, "
          f"{'stochastic' if args.stochastic else 'deterministic'}")
    print(f"{'='*70}")
    print(f"  harvested_mg   median={stats['median_harvested_mg']:.1f}  "
          f"p25={stats['p25_harvested_mg']:.1f}   (gate: median>={target.get('min_median_harvested_mg')}  "
          f"p25>={target.get('min_p25_harvested_mg')})")
    print(f"  time_avg_od    median={stats['median_time_avg_od']:.4f}   "
          f"(gate: median>={target.get('min_median_time_avg_od')})")
    print(f"  crash_rate     {stats['crash_rate']*100:.1f}%   (gate: <={target.get('max_crash_rate', 0)*100:.0f}%)")

    passed = (
        stats["median_harvested_mg"] >= target.get("min_median_harvested_mg", 1e9)
        and stats["p25_harvested_mg"] >= target.get("min_p25_harvested_mg", 1e9)
        and stats["crash_rate"] <= target.get("max_crash_rate", 0)
        and stats["median_time_avg_od"] >= target.get("min_median_time_avg_od", 1e9)
    )
    print(f"\n  HOLDS ON HELD-OUT SAMPLE: {'YES' if passed else 'NO'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
