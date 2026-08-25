"""
td3_held_out_sweep.py — independent held-out validation for the TD3+BC actor (v35),
mirroring diagnostics/held_out_sweep.py's role for PPO: 40 cold-start episodes on the
SAME distribution the curriculum gate uses (90% lognormal(100,400), 10% adversarial
30-80), deterministic policy, scored against the D2 gate.

Why this is needed even though v35's in-training det-eval already looked like a D2 pass:
this project's own methodology (novelty_report.md's C3) exists specifically because
in-training eval has produced false positives before (v14/v17/v26/TD-MPC2 v27 all passed
in-training and failed held-out). v35's in-training det-eval used only 12-18 episodes per
chunk from curriculum_schedule's own sampling — this sweep is the same held-out
methodology as every other checkpoint in finalresults.md, not a different, easier bar.

Usage (from repo root, PPO_IBM/):
    python experiments/bc_scaffold/scripts/td3_held_out_sweep.py --n 40
"""

import os
import sys
import argparse

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments"), os.path.join(ROOT, "legacy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from genetic_env import GeneticPhotobioreactorEnv
from TD3 import RecurrentActor, OBS_DIM, ACTION_DIM, MAX_CELLS, DEVICE

GATE = {"harvest": 90.0, "p25": 50.0, "crash": 0.08, "time_od": 0.011}


def sample_init_cells(rng, adversarial_frac=0.10):
    if rng.rand() < adversarial_frac:
        return int(rng.uniform(30, 80))
    return int(np.exp(rng.uniform(np.log(100), np.log(400))))


def run_episode(actor, difficulty, init_cells, seed):
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    hidden = actor.initial_hidden(batch=1)
    done, step, info = False, 0, {}
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
            action_t, hidden = actor(obs_t, hidden)
            action = action_t.view(-1).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
    return {
        "seed": seed, "init_cells": init_cells, "steps": step, "crashed": step < env.max_steps,
        "harvested_mg": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-path", default="model_data/td3_checkpoints/actor.pth")
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    actor = RecurrentActor(OBS_DIM, ACTION_DIM).to(DEVICE)
    actor.load_state_dict(torch.load(args.actor_path, map_location=DEVICE))
    actor.eval()

    rng = np.random.RandomState(args.base_seed)
    results = []
    n_adv = 0
    for i in range(args.n):
        seed = args.base_seed + i
        init_cells = sample_init_cells(rng)
        if init_cells <= 80:
            n_adv += 1
        r = run_episode(actor, args.difficulty, init_cells, seed)
        results.append(r)
        tag = "ADV" if init_cells <= 80 else "   "
        print(f"  [{i+1:3d}/{args.n}] seed={seed:5d} {tag} init={init_cells:5d}  "
              f"steps={r['steps']:5d}  crashed={r['crashed']!s:5}  "
              f"harvested={r['harvested_mg']:7.1f}mg  time_avg_od={r['time_avg_od']:.4f}")

    harvested = np.array([r["harvested_mg"] for r in results])
    time_od = np.array([r["time_avg_od"] for r in results])
    crash_rate = float(np.mean([r["crashed"] for r in results]))
    med_h, p25_h, med_od = float(np.median(harvested)), float(np.percentile(harvested, 25)), float(np.median(time_od))

    print(f"\n{'='*70}")
    print(f"  TD3+BC HELD-OUT SWEEP  (D{args.difficulty}, n={args.n}, {n_adv} adversarial cold starts)")
    print(f"{'='*70}")
    print(f"  crash_rate           : {crash_rate*100:.1f}%")
    print(f"  harvested_mg  median : {med_h:.1f}   p25: {p25_h:.1f}   min: {harvested.min():.1f}   max: {harvested.max():.1f}")
    print(f"  time_avg_od   median : {med_od:.4f}   p25: {np.percentile(time_od,25):.4f}")
    print(f"\n  vs D2 curriculum gate: harvest>={GATE['harvest']} p25>={GATE['p25']} "
          f"crash<={GATE['crash']*100:.0f}% time_od>={GATE['time_od']}")
    ok = (med_h >= GATE["harvest"] and p25_h >= GATE["p25"]
          and crash_rate <= GATE["crash"] and med_od >= GATE["time_od"])
    print(f"  holds on held-out sample: {'YES' if ok else 'NO'}")


if __name__ == "__main__":
    main()
