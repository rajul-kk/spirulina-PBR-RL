"""Trained policy vs scripted expert across cold-start population, incl. the 500-5000
(full rationale: docs/decision_history.md#--experiments-env_diagnosis-population_range_check-py-1)"""

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
from TD3 import RecurrentActor, OBS_DIM, ACTION_DIM, MAX_CELLS, DEVICE, HIDDEN_RESET_INTERVAL
from actor_io import load_actor

EXPERT_OD_SETPOINT = 0.015
EXPERT_FRAC_CAP = 0.30


def expert_action(env):
    surplus = (float(env.od) / EXPERT_OD_SETPOINT) - 1.0
    frac = float(np.clip(surplus, 0.0, EXPERT_FRAC_CAP))
    f_max = float(getattr(env, "F_MAX", 0.5))
    return np.array([
        np.interp(70.0, [50, 200], [-1, 1]),
        np.interp(950.0, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def episode(actor, difficulty, init_cells, seed):
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    hidden = actor.initial_hidden(batch=1) if actor is not None else None
    since_reset = 0
    done, info = False, {}
    with torch.no_grad():
        while not done:
            if actor is None:
                a = expert_action(env)
            else:
                if since_reset >= HIDDEN_RESET_INTERVAL:
                    hidden = actor.initial_hidden(batch=1)
                    since_reset = 0
                t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
                at, hidden = actor(t, hidden)
                a = at.view(-1).cpu().numpy()
                since_reset += 1
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
    return {
        "harvest": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
        "crashed": env.step_count < env.max_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-path", default="model_data/archive_v45_td3_hidden_reset/td3_checkpoints/actor.pth")
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()

    actor, core = load_actor(args.actor_path, OBS_DIM, ACTION_DIM, DEVICE)
    print(f"  actor: {args.actor_path}  core={core}")

    pops = [150, 300, 500, 1000, 2000, 3500, 5000]
    print("=" * 86)
    print(f"  COLD-START POPULATION SWEEP (D{args.difficulty}, {args.seeds} seeds each)")
    print(f"  policy: {args.actor_path}")
    print("=" * 86)
    print(f"  {'init':>6} | {'POLICY harvest':>15}{'od':>9}{'crash%':>8} | "
          f"{'EXPERT harvest':>15}{'od':>9}{'crash%':>8} | {'ratio':>7}")
    print("  " + "-" * 82)
    for p in pops:
        pol = [episode(actor, args.difficulty, p, 3000 + i) for i in range(args.seeds)]
        exp = [episode(None, args.difficulty, p, 3000 + i) for i in range(args.seeds)]
        ph = np.median([r["harvest"] for r in pol])
        eh = np.median([r["harvest"] for r in exp])
        po = np.median([r["time_avg_od"] for r in pol])
        eo = np.median([r["time_avg_od"] for r in exp])
        pc = 100 * np.mean([r["crashed"] for r in pol])
        ec = 100 * np.mean([r["crashed"] for r in exp])
        ratio = ph / eh if eh > 1e-9 else float("nan")
        print(f"  {p:>6} | {ph:>15.1f}{po:>9.4f}{pc:>8.0f} | {eh:>15.1f}{eo:>9.4f}{ec:>8.0f} | {ratio:>7.2f}")

    print("\n  ratio = policy harvest / expert harvest on the SAME start (1.00 = matches expert)")
    print("  NOTE: the held-out sweep samples lognormal(100,400) + 10% adversarial, so it")
    print("        never tests starts above ~400 cells. Rows >=500 are untested territory.")


if __name__ == "__main__":
    main()
