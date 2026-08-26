"""
expert_sweep.py — runs the non-learned proportional harvest law (same as
bc/bc_pretrain.py) directly against genetic_env.py, zero NN and zero training, on
the curriculum's own held-out cold-start distribution. Isolates whether PPO/TD-MPC2's
time_avg_od bottleneck is environment difficulty or an RL-discovery problem.

Usage (from repo root, PPO_IBM/):
    python experiments/bc_scaffold/scripts/expert_sweep.py --n 40 --difficulty 0
    python experiments/bc_scaffold/scripts/expert_sweep.py --n 40 --difficulty 1
    python experiments/bc_scaffold/scripts/expert_sweep.py --n 40 --difficulty 2
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import csv
import numpy as np

from genetic_env import GeneticPhotobioreactorEnv

# Same law/constants as bc/bc_pretrain.py, not re-tuned here.
EXPERT_STIR_RANGE = (60.0, 80.0)
EXPERT_LIGHT_RANGE = (900.0, 1000.0)
EXPERT_OD_SETPOINT = 0.015
EXPERT_GAIN = 1.0
EXPERT_FRAC_CAP = 0.30

GATES = {
    0: {"harvest": 30.0, "p25": 15.0, "crash": 0.15, "time_od": 0.004},
    1: {"harvest": 60.0, "p25": 30.0, "crash": 0.10, "time_od": 0.008},
    2: {"harvest": 90.0, "p25": 50.0, "crash": 0.08, "time_od": 0.011},
}


def sample_init_cells(rng, adversarial_frac=0.10):
    if rng.rand() < adversarial_frac:
        return int(rng.uniform(30, 80))
    return int(np.exp(rng.uniform(np.log(100), np.log(400))))


def expert_harvest_frac(od):
    surplus = (float(od) / EXPERT_OD_SETPOINT) - 1.0
    return float(np.clip(EXPERT_GAIN * surplus, 0.0, EXPERT_FRAC_CAP))


def run_episode(difficulty, init_cells, seed, stir, light):
    env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
    env.reset(seed=seed)
    f_max = float(getattr(env, "F_MAX", 0.5))

    done = False
    step = 0
    info = {}
    while not done:
        frac = expert_harvest_frac(getattr(env, "od", 0.0))
        action = np.array([
            np.interp(stir, [50, 200], [-1, 1]),
            np.interp(light, [0, 2000], [-1, 1]),
            np.interp(frac, [0, f_max], [-1, 1]),
        ], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    crashed = step < env.max_steps
    return {
        "seed": seed, "init_cells": init_cells, "steps": step, "crashed": crashed,
        "harvested_mg": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--out", type=str, default=None,
                    help="CSV path to write per-episode results (default: results/expert_sweep_D{d}.csv)")
    args = ap.parse_args()

    rng = np.random.RandomState(args.base_seed)
    ctrl_rng = np.random.RandomState(args.base_seed + 999)  # separate stream for stir/light draw
    stir = float(ctrl_rng.uniform(*EXPERT_STIR_RANGE))
    light = float(ctrl_rng.uniform(*EXPERT_LIGHT_RANGE))

    print(f"Scripted expert, no NN, no training. stir={stir:.1f} light={light:.1f} "
          f"harvest=clip({EXPERT_GAIN}*(od/{EXPERT_OD_SETPOINT}-1), 0, {EXPERT_FRAC_CAP})")
    print(f"D{args.difficulty}, n={args.n} held-out cold starts (90% lognormal(100,400), 10% adversarial 30-80)\n")

    results = []
    for i in range(args.n):
        seed = args.base_seed + i
        init_cells = sample_init_cells(rng)
        r = run_episode(args.difficulty, init_cells, seed, stir, light)
        results.append(r)
        tag = "ADV" if init_cells <= 80 else "   "
        print(f"  [{i+1:3d}/{args.n}] seed={seed:5d} {tag} init={init_cells:5d}  "
              f"steps={r['steps']:5d}  crashed={r['crashed']!s:5}  "
              f"harvested={r['harvested_mg']:7.1f}mg  time_avg_od={r['time_avg_od']:.4f}")

    harvested = np.array([r["harvested_mg"] for r in results])
    time_od = np.array([r["time_avg_od"] for r in results])
    crash_rate = float(np.mean([r["crashed"] for r in results]))
    med_h, p25_h = float(np.median(harvested)), float(np.percentile(harvested, 25))
    med_od = float(np.median(time_od))

    gate = GATES[args.difficulty]
    ok = (med_h >= gate["harvest"] and p25_h >= gate["p25"]
          and crash_rate <= gate["crash"] and med_od >= gate["time_od"])

    print(f"\n{'='*70}")
    print(f"  SCRIPTED EXPERT, NO LEARNING  (D{args.difficulty}, n={args.n})")
    print(f"{'='*70}")
    print(f"  crash_rate           : {crash_rate*100:.1f}%   [gate <= {gate['crash']*100:.0f}%]")
    print(f"  harvested_mg  median : {med_h:.1f}   p25: {p25_h:.1f}   [gate: median>={gate['harvest']} p25>={gate['p25']}]")
    print(f"  time_avg_od   median : {med_od:.4f}   [gate >= {gate['time_od']}]")
    print(f"  holds on held-out sample: {'YES' if ok else 'NO'}")

    out_path = args.out or _os.path.join(_os.path.dirname(__file__), "..", "results", f"expert_sweep_D{args.difficulty}.csv")
    out_path = _os.path.abspath(out_path)
    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "init_cells", "steps", "crashed", "harvested_mg", "time_avg_od"])
        w.writeheader()
        w.writerows(results)
    print(f"\n  Wrote per-episode results to {out_path}")


if __name__ == "__main__":
    main()
