"""Hidden-state dynamics of a TD3 recurrent actor, closed loop on --split test episodes.
Per step records L2 norm of the recurrent state; LSTM: max|c| and frac(|tanh c|>0.95); GRU/LRU: max|h|.
Runs modes 'reset600' and 'never' (reset 1000000), reusing harness.run_episode's env construction.
usage (from experiments/program_control):
  python diagnostics/state_dynamics.py --actor <abs actor.pth> --name lru600_s3 [--episodes 3] [--max-steps N]
Writes diagnostics/state_dynamics/<name>.csv and <name>_summary.txt"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PC = os.path.dirname(HERE)
sys.path.insert(0, PC)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(PC)), "td3"))
sys.path.insert(0, os.path.dirname(PC))
import torch  # noqa: E402

torch.set_num_threads(1)
import harness  # noqa: E402
from actor_io import load_actor  # noqa: E402

MODES = {"reset600": 600, "never": 1_000_000}
CHECK = (60, 300, 600, 1200)


def state_stats(hidden, core):
    if core == "lstm":
        h, c = hidden
        return dict(norm=float(torch.sqrt((h ** 2).sum() + (c ** 2).sum())), max_c=float(c.abs().max()),
                    sat_frac=float((torch.tanh(c).abs() > 0.95).float().mean()))
    return dict(norm=float(hidden.norm()), max_h=float(hidden.abs().max()))


def run(actor, core, reset, init_cells, seed, max_steps=None):
    from genetic_env import GeneticPhotobioreactorEnv
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=harness.MAX_CELLS, initial_cells=init_cells, difficulty=2)
    obs, _ = env.reset(seed=seed)
    f_max = float(env.F_MAX)
    hidden, since, t, done, rows = actor.initial_hidden(batch=1), 0, 0, False, []
    while not done and (max_steps is None or t < max_steps):
        if since >= reset:
            hidden, since = actor.initial_hidden(batch=1), 0
        x = torch.tensor(obs[:6], dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            a, hidden = actor(x, hidden)
        a = a.view(-1).numpy()
        since += 1
        rows.append(dict(t=t, **state_stats(hidden, core)))
        act = harness.to_env_action(np.interp(a[0], [-1, 1], [50, 200]), np.interp(a[1], [-1, 1], [0, 2000]),
                                    np.interp(a[2], [-1, 1], [0, 0.5]), f_max)
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
        t += 1
    return rows, float(env.cumulative_harvested_mg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--outdir", default=os.path.join(HERE, "state_dynamics"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    actor, core = load_actor(a.actor, device=torch.device("cpu"))
    jobs = harness.split_jobs("test", 2)[:a.episodes]   # test split only
    allrows = []
    for mode, reset in MODES.items():
        for ep, (ic, seed) in enumerate(jobs):
            rows, harv = run(actor, core, reset, ic, seed, a.max_steps)
            for r in rows:
                allrows.append(dict(mode=mode, episode=ep, seed=seed, harvested_mg=harv, **r))
            print(f"{a.name} {mode} ep{ep} seed={seed} steps={len(rows)} harvest={harv:.0f}", flush=True)
    keys = ["mode", "episode", "seed", "t", "norm", "max_c", "sat_frac", "max_h", "harvested_mg"]
    with open(os.path.join(a.outdir, a.name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(allrows)
    lines = [f"actor {a.name} core={core} path={a.actor}", f"episodes={len(jobs)} (test split, first {len(jobs)})",
             "state L2 norm (mean over episodes) at step 60/300/600/1200/end; peak saturation (LSTM) / peak max|c| or |h|"]
    for mode in MODES:
        eps = sorted({r["episode"] for r in allrows if r["mode"] == mode})
        by = {e: [r for r in allrows if r["mode"] == mode and r["episode"] == e] for e in eps}
        cols = []
        for s in list(CHECK) + ["end"]:
            v = [(b[-1] if s == "end" else b[s - 1])["norm"] for b in by.values() if s == "end" or len(b) >= s]
            cols.append(f"{s}:{np.mean(v):.2f}" if v else f"{s}:n/a")
        peak = {}
        for k in ("sat_frac", "max_c", "max_h"):
            v = [r[k] for r in allrows if r["mode"] == mode and k in r]
            if v:
                peak[k] = max(v)
        lines.append(f"[{mode}] norm@ " + "  ".join(cols) + "  peak " +
                     "  ".join(f"{k}={v:.3f}" for k, v in peak.items()))
    txt = "\n".join(lines)
    open(os.path.join(a.outdir, a.name + "_summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
