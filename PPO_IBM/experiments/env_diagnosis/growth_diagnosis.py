"""Is the yield real productivity, or is the culture being mined down?
(full rationale: docs/decision_history.md#--experiments-env_diagnosis-growth_diagnosis-py-1)"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments"), os.path.join(ROOT, "legacy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from genetic_env import GeneticPhotobioreactorEnv
from TD3 import (OBS_DIM, ACTION_DIM, MAX_CELLS, DEVICE, HIDDEN_RESET_INTERVAL,
                 expert_harvest_frac, EXPERT_STIR_RANGE, EXPERT_LIGHT_RANGE, EXPERT_FRAC_CAP)
from actor_io import load_actor

OD_TARGET = 0.012


def _mass_mg(env):
    return float(np.sum(env.cells_mass[env._aidx()]) * 1e-9)


def run(policy, init_cells, seed, difficulty=2):
    """policy: 'expert' or a loaded actor. Returns growth telemetry."""
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    actor = None if policy == "expert" else policy
    hid = actor.initial_hidden(1) if actor is not None else None
    since = 0

    pops, ods, masses = [], [], []
    prev_mass = _mass_mg(env)
    gross_growth = 0.0          # sum of positive mass deltas: biomass actually produced
    harvested_prev = 0.0
    growth_before, growth_after = 0.0, 0.0

    with torch.no_grad():
        while True:
            if actor is None:
                stir = float(np.mean(EXPERT_STIR_RANGE)); light = float(np.mean(EXPERT_LIGHT_RANGE))
                act = np.array([np.interp(stir, [50, 200], [-1, 1]),
                                np.interp(light, [0, 2000], [-1, 1]),
                                np.interp(expert_harvest_frac(env.od), [0.0, env.F_MAX], [-1, 1])],
                               dtype=np.float32)
            else:
                if since >= HIDDEN_RESET_INTERVAL:
                    hid = actor.initial_hidden(1); since = 0
                ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
                at, hid = actor(ot, hid)
                act = at.view(-1).cpu().numpy()
                since += 1

            obs, _, term, trunc, info = env.step(act)
            m = _mass_mg(env)
            h = float(info.get("cumulative_harvested_mg", 0.0))
            # Growth = mass gained excluding the drop caused by this step's harvest.
            delta = (m - prev_mass) + (h - harvested_prev)
            if delta > 0:
                gross_growth += delta
                if env.step_count < 3600:
                    growth_before += delta
                else:
                    growth_after += delta
            prev_mass, harvested_prev = m, h
            pops.append(env.num_active); ods.append(env.od); masses.append(m)
            if term or trunc:
                break

    pops, ods = np.array(pops), np.array(ods)
    return {
        "init": init_cells, "steps": env.step_count, "crashed": env.step_count < env.max_steps,
        "harvest": float(info.get("cumulative_harvested_mg", 0.0)),
        "harvest_bh": float(info.get("harvested_mg_back_half", 0.0)),
        "pop_final": int(env.num_active), "pop_peak": int(pops.max()), "pop_min": int(pops.min()),
        "od_med": float(np.median(ods)), "od_peak": float(ods.max()), "od_final": float(ods[-1]),
        "mass_start": float(masses[0]), "mass_final": float(masses[-1]),
        "growth": gross_growth, "growth_1st": growth_before, "growth_2nd": growth_after,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", default="380,1500,4000")
    ap.add_argument("--seed", type=int, default=900_003)
    ap.add_argument("--difficulty", type=int, default=2)
    args = ap.parse_args()
    starts = [int(s) for s in args.starts.split(",")]

    pols = []
    for tag, path in (("v50 LRU best", "model_data/td3_lru_checkpoints_best/actor.pth"),
                      ("v49 LSTM final", "model_data/td3_checkpoints/actor.pth")):
        if os.path.exists(path):
            a, core = load_actor(path, OBS_DIM, ACTION_DIM, DEVICE)
            pols.append((f"{tag} ({core})", a))
    pols.append(("scripted expert", "expert"))

    print("=" * 108)
    print(f"  GROWTH DIAGNOSIS  D{args.difficulty}  seed={args.seed}   (OD target {OD_TARGET})")
    print("=" * 108)
    for init in starts:
        print(f"\n  init={init} cells")
        print(f"    {'policy':<22} {'harvest':>8} {'growth':>8} {'h/g':>6} {'pop i->f':>14} "
              f"{'OD med':>7} {'OD pk':>7} {'mass i->f':>15} {'growth 1st/2nd':>15}")
        for tag, pol in pols:
            r = run(pol, init, args.seed, args.difficulty)
            ratio = r["harvest"] / max(r["growth"], 1e-9)
            split = f"{r['growth_1st']:.0f}/{r['growth_2nd']:.0f}"
            print(f"    {tag:<22} {r['harvest']:8.1f} {r['growth']:8.1f} {ratio:6.2f} "
                  f"{r['init']:6d}->{r['pop_final']:<7d} {r['od_med']:7.4f} {r['od_peak']:7.4f} "
                  f"{r['mass_start']:6.1f}->{r['mass_final']:<8.1f} {split:>15}"
                  + ("  CRASH" if r["crashed"] else ""))
    print("\n  h/g = harvested / biomass grown. ~1.0 means yield came from genuine production;")
    print("  >1 means the starting culture was drawn down; <1 means growth outpaced harvesting.")


if __name__ == "__main__":
    sys.exit(main())
