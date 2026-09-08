"""What did the LRU actually learn to remember, and does its state stay bounded?
(full rationale: docs/decision_history.md#--experiments-env_diagnosis-lru_memory_check-py-1)"""

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
from TD3 import OBS_DIM, ACTION_DIM, MAX_CELLS, DEVICE, SEQ_LEN
from actor_io import load_actor, detect_core

HARVEST_INTERVAL = 600


def horizons(actor):
    """Per-channel decay lam and effective memory horizon 1/(1-lam), in steps."""
    nu_log = actor.lstm.nu_log.detach().cpu().numpy()
    lam = np.exp(-np.exp(nu_log))
    return lam, 1.0 / np.maximum(1.0 - lam, 1e-12)


def report_spectrum(tag, lam, h):
    qs = [0, 5, 25, 50, 75, 95, 100]
    pc = np.percentile(h, qs)
    print(f"  {tag}")
    print(f"    lam      min={lam.min():.6f}  median={np.median(lam):.6f}  max={lam.max():.6f}")
    print("    horizon  " + "  ".join(f"p{q}={v:,.0f}" for q, v in zip(qs, pc)))
    for thr, label in ((SEQ_LEN, f"> train window ({SEQ_LEN})"),
                       (HARVEST_INTERVAL, f"> harvest interval ({HARVEST_INTERVAL})")):
        n = int((h > thr).sum())
        print(f"    channels {label:<32}: {n:3d}/{len(h)} ({100*n/len(h):.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-path", default="model_data/td3_lru_checkpoints_best/actor.pth")
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--init-cells", type=int, default=300)
    ap.add_argument("--seed", type=int, default=900_001)
    ap.add_argument("--no-reset", action="store_true",
                    help="run the rollout WITHOUT the periodic hidden reset, to test boundedness")
    args = ap.parse_args()

    actor, core = load_actor(args.actor_path, OBS_DIM, ACTION_DIM, DEVICE)
    if core != "lru":
        print(f"ERROR: {args.actor_path} is a '{core}' checkpoint; this diagnostic is LRU-only.")
        return 1

    print("=" * 78)
    print(f"  LRU MEMORY CHECK  {args.actor_path}")
    print("=" * 78)

    # 1. What the trained decay spectrum looks like, against a fresh init as control.
    from TD3_lru import LRUActor
    torch.manual_seed(0)
    fresh = LRUActor(OBS_DIM, ACTION_DIM)
    lam_f, h_f = horizons(fresh)
    lam_t, h_t = horizons(actor)
    print("\nDecay spectrum (lam = exp(-exp(nu_log)); horizon = 1/(1-lam) steps)")
    report_spectrum("at init (control):", lam_f, h_f)
    print()
    report_spectrum("trained:", lam_t, h_t)
    moved = float(np.mean(np.abs(h_t - h_f) / np.maximum(h_f, 1e-9)))
    print(f"\n    mean |relative| horizon change vs init: {100*moved:.1f}%")
    print("    (near 0% would mean the decay parameters barely trained, and the memory")
    print("     claim for the LRU rests on the initialisation rather than on learning)")

    # 2. Does |h| stay bounded over a full episode with no reset?
    np.random.seed(args.seed)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=args.init_cells,
                                    difficulty=args.difficulty)
    obs, _ = env.reset(seed=args.seed)
    hid = actor.initial_hidden(1)
    mags, steps_since = [], 0
    with torch.no_grad():
        while True:
            if not args.no_reset and steps_since >= SEQ_LEN:
                hid = actor.initial_hidden(1)
                steps_since = 0
            ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
            at, hid = actor(ot, hid)
            mags.append(float(hid.abs().max()))
            obs, _, term, trunc, info = env.step(at.view(-1).cpu().numpy())
            steps_since += 1
            if term or trunc:
                break
    m = np.array(mags)
    mode = "NO reset (free-running)" if args.no_reset else f"reset every {SEQ_LEN} steps"
    print(f"\nState magnitude over {len(m)} steps, {mode}")
    for s in (10, 100, 1000, 3000, len(m) - 1):
        if s < len(m):
            print(f"    |h| at step {s:5d}: {m[s]:8.3f}")
    print(f"    max={m.max():.3f}  mean={m.mean():.3f}")
    first, last = m[:len(m) // 2], m[len(m) // 2:]
    drift = (last.mean() - first.mean()) / max(first.mean(), 1e-9)
    print(f"    2nd-half vs 1st-half mean drift: {100*drift:+.1f}%  "
          f"({'bounded' if abs(drift) < 0.5 else 'GROWING -- investigate'})")
    print(f"\n    episode: harvested={info.get('cumulative_harvested_mg', 0.0):.1f}mg  "
          f"crashed={env.step_count < env.max_steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
