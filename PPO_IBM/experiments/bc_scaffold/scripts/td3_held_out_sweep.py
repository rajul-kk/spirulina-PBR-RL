"""td3_held_out_sweep.py — independent held-out validation for a TD3 actor checkpoint, ...
(full rationale: docs/decision_history.md#--experiments-bc_scaffold-scripts-td3_held_out_sweep-py-1)"""

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
    np.random.seed(seed)
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
        "harvested_mg_back_half": float(info.get("harvested_mg_back_half", 0.0)),
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

    # Item 3: adversarial starts (<=80 cells) are scored on SURVIVAL, not yield -- the
    # (full rationale: docs/decision_history.md#--td3_held_out_sweep-adversarial-survival)
    adv = [r for r in results if r["init_cells"] <= 80]
    yielding = [r for r in results if r["init_cells"] > 80]

    harvested = np.array([r["harvested_mg"] for r in yielding])
    time_od = np.array([r["time_avg_od"] for r in yielding])
    crash_rate = float(np.mean([r["crashed"] for r in results]))  # crash over ALL episodes
    med_h, p25_h, med_od = float(np.median(harvested)), float(np.percentile(harvested, 25)), float(np.median(time_od))
    cvar_cutoff = np.percentile(harvested, 10)
    tail = harvested[harvested <= cvar_cutoff]
    cvar10 = float(np.mean(tail)) if len(tail) > 0 else float(harvested.min())

    print(f"\n{'='*70}")
    print(f"  TD3+BC HELD-OUT SWEEP  (D{args.difficulty}, n={args.n}, {n_adv} adversarial cold starts)")
    print(f"{'='*70}")
    print(f"  crash_rate (all eps) : {crash_rate*100:.1f}%")
    print(f"  yield scored on {len(yielding)} non-adversarial episodes:")
    print(f"  harvested_mg  median : {med_h:.1f}   p25: {p25_h:.1f}   cvar10: {cvar10:.1f}   min: {harvested.min():.1f}   max: {harvested.max():.1f}")
    print(f"  time_avg_od   median : {med_od:.4f}   p25: {np.percentile(time_od,25):.4f}")
    bh = np.array([r["harvested_mg_back_half"] for r in yielding])
    print(f"  harvest window: full-episode median={med_h:.1f}  "
          f"back-half median={np.median(bh):.1f}  "
          f"back-half share={100*np.median(bh)/max(med_h,1e-9):.0f}%  "
          f"(back-half p25={np.percentile(bh,25):.1f})")

    if adv:
        adv_crash = 100 * float(np.mean([r["crashed"] for r in adv]))
        print(f"  adversarial ({len(adv)} eps, survival-scored): crash_rate={adv_crash:.1f}%  "
              f"harvest(not gated)={np.median([r['harvested_mg'] for r in adv]):.1f}")

    # Item 2: per-bucket breakdown -- a pooled median hides large within-range spread.
    # (full rationale: docs/decision_history.md#--td3_held_out_sweep-per-bucket)
    print("\n  per-bucket (pooled medians hide within-range spread):")
    buckets = [(0, 81, "adversarial <=80"), (81, 200, "low 81-200"), (200, 400, "low 200-400"),
               (400, 1500, "mid 400-1500"), (1500, 10 ** 9, "high >1500")]
    for lo, hi, lab in buckets:
        v = [r for r in results if lo <= r["init_cells"] < hi]
        if not v:
            continue
        hv = [r["harvested_mg"] for r in v]
        cv = 100 * float(np.mean([r["crashed"] for r in v]))
        print(f"    {lab:<18} n={len(v):2d}  median={np.median(hv):7.1f}  "
              f"min={min(hv):6.1f}  max={max(hv):7.1f}  crash={cv:.0f}%")

    print(f"\n  vs D2 curriculum gate: harvest>={GATE['harvest']} p25>={GATE['p25']} "
          f"crash<={GATE['crash']*100:.0f}% time_od>={GATE['time_od']}")
    ok = (med_h >= GATE["harvest"] and p25_h >= GATE["p25"]
          and crash_rate <= GATE["crash"] and med_od >= GATE["time_od"])
    print(f"  holds on held-out sample: {'YES' if ok else 'NO'}")
    print("  NOTE: this sweep samples lognormal(100,400)+10% adversarial, so it does NOT")
    print("        test the 600-5000 range that training samples. Use population_range_check.py.")


if __name__ == "__main__":
    main()
