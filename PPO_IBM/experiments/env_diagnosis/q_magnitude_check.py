"""Is the high-population regression critic divergence or actor policy drift?
(full rationale: docs/decision_history.md#--experiments-env_diagnosis-q_magnitude_check-py-1)"""

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
from TD3 import OBS_DIM, ACTION_DIM, MAX_CELLS, DEVICE, SEQ_LEN, GAMMA, HIDDEN_RESET_INTERVAL
from curriculum_schedule import det_eval_set
from actor_io import load_actor, load_critic


def rollout(actor, difficulty, init_cells, seed):
    """Deterministic rollout matching run_td3_eval_episode, keeping obs/act/rew."""
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    hid = actor.initial_hidden(1)
    O, A, R = [], [], []
    since = 0
    with torch.no_grad():
        while True:
            if since >= HIDDEN_RESET_INTERVAL:
                hid = actor.initial_hidden(1)
                since = 0
            ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
            at, hid = actor(ot, hid)
            act = at.view(-1).cpu().numpy()
            O.append(obs); A.append(act)
            obs, rew, term, trunc, info = env.step(act)
            R.append(float(rew))
            since += 1
            if term or trunc:
                break
    return (np.array(O, dtype=np.float32), np.array(A, dtype=np.float32),
            np.array(R, dtype=np.float64), info, env.step_count < env.max_steps)


def returns_to_go(rew, gamma):
    G = np.empty_like(rew)
    acc = 0.0
    for i in range(len(rew) - 1, -1, -1):
        acc = rew[i] + gamma * acc
        G[i] = acc
    return G


def critic_windows(critic, O, A):
    """Q at the first step of each SEQ_LEN window, from a zero state -- exactly the
    conditioning the critic was trained under. Evaluating one 7200-step sequence instead
    would be out of distribution and would not measure what training actually optimised."""
    q1s, q2s, starts = [], [], []
    with torch.no_grad():
        for s in range(0, len(O) - SEQ_LEN + 1, SEQ_LEN):
            o = torch.tensor(O[s:s + SEQ_LEN], device=DEVICE).unsqueeze(0)
            a = torch.tensor(A[s:s + SEQ_LEN], device=DEVICE).unsqueeze(0)
            q1, q2, _, _ = critic(o, a)
            q1s.append(float(q1[0, 0, 0])); q2s.append(float(q2[0, 0, 0])); starts.append(s)
    return np.array(q1s), np.array(q2s), np.array(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="model_data/td3_lru_checkpoints")
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--adversarial-max", type=int, default=80)
    args = ap.parse_args()

    actor, core = load_actor(os.path.join(args.ckpt_dir, "actor.pth"), OBS_DIM, ACTION_DIM, DEVICE)
    critic, _ = load_critic(os.path.join(args.ckpt_dir, "critic.pth"), OBS_DIM, ACTION_DIM, DEVICE)

    print("=" * 96)
    print(f"  Q-MAGNITUDE CHECK  {args.ckpt_dir}  core={core}  D{args.difficulty}  gamma={GAMMA}")
    print("=" * 96)
    print(f"  {'init':>6} {'harvest':>9} {'MC return':>11} {'Q_min':>11} {'gap':>11} {'gap%':>8} "
          f"{'|q1-q2|':>9} {'crash':>6}")
    print("  " + "-" * 92)

    rows = []
    for init_cells, seed in det_eval_set():
        O, A, R, info, crashed = rollout(actor, args.difficulty, init_cells, seed)
        G = returns_to_go(R, GAMMA)
        q1, q2, starts = critic_windows(critic, O, A)
        qmin = np.minimum(q1, q2)                 # TD3 uses the twin minimum
        Gw = G[starts]
        gap = qmin - Gw
        rel = 100.0 * np.median(gap) / max(abs(np.median(Gw)), 1e-9)
        rows.append((init_cells, float(info.get("cumulative_harvested_mg", 0.0)),
                     float(np.median(Gw)), float(np.median(qmin)), float(np.median(gap)), rel,
                     float(np.median(np.abs(q1 - q2))), crashed))
        print(f"  {init_cells:6d} {rows[-1][1]:9.1f} {rows[-1][2]:11.2f} {rows[-1][3]:11.2f} "
              f"{rows[-1][4]:11.2f} {rel:7.0f}% {rows[-1][6]:9.3f} {str(crashed):>6}")

    yielding = [r for r in rows if r[0] > args.adversarial_max]
    low = [r for r in yielding if r[0] < 1500]
    high = [r for r in yielding if r[0] >= 1500]
    print("\n  Interpretation")
    for tag, grp in (("low/mid (<1500 cells)", low), ("high (>=1500 cells)", high)):
        if not grp:
            continue
        print(f"    {tag:<24} median MC return={np.median([r[2] for r in grp]):8.2f}  "
              f"Q={np.median([r[3] for r in grp]):8.2f}  "
              f"gap={np.median([r[4] for r in grp]):8.2f} ({np.median([r[5] for r in grp]):.0f}%)  "
              f"|q1-q2|={np.median([r[6] for r in grp]):.3f}")
    if low and high:
        gl, gh = np.median([r[5] for r in low]), np.median([r[5] for r in high])
        print(f"\n    Relative overestimation is {gh:.0f}% at high vs {gl:.0f}% at low population.")
        print("    A much larger gap at high population indicates CRITIC DIVERGENCE in the")
        print("    high-return regime. A similar gap in both, with high-population harvest")
        print("    still depressed, would instead indicate ACTOR POLICY DRIFT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
