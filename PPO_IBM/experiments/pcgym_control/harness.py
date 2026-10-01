"""Score a controller program on a PC-Gym task (docs/reports/pcgym_protocol.md).

A controller is a Python file defining `class Controller` with `__init__(self, params=None)` and
`act(self, obs) -> action`; a fresh instance is built for every episode. It may import only
numpy, math and collections.

Splits (seeds are disjoint by construction):
  search  12 episodes, seeds 3,000,000+     the only split CMA-ES sees
  final   200 episodes, seeds 20,000,000+   run once per frozen controller
Pilot batches drawn by plant_trial.py use seeds 5,000,000+ in a block per run.

  python harness.py controllers/pid.py --task cstr --split final --out results/final/x.json
"""
import argparse
import ast
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # workers import tasks.py from here
ALLOWED_IMPORTS = {"numpy", "math", "collections"}
SPLITS = {"search": (3_000_000, 12), "final": (20_000_000, 200)}


def check_source(path):
    """Reject controllers that import anything beyond the allowed modules or touch files."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        bad = [m for m in mods if m not in ALLOWED_IMPORTS]
        if bad:
            raise ValueError(f"disallowed import: {bad}")
        if isinstance(node, ast.Name) and node.id in {"open", "exec", "eval", "__import__", "compile"}:
            raise ValueError(f"disallowed call: {node.id}")


def load_controller(path, params=None):
    spec = importlib.util.spec_from_file_location("ctrl_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Controller(params) if params is not None else mod.Controller()


def run_episode(job):
    path, task, seed, params, privileged, want_trace = job
    from tasks import TASKS
    try:
        ctrl = load_controller(path, params)
    except Exception as e:
        class _Dead:
            def act(self, obs, _e=e):
                raise RuntimeError(f"{type(_e).__name__} at load: {_e}")
        ctrl = _Dead()
    summary, trace = TASKS[task].run(ctrl, seed, privileged)
    return (summary, trace) if want_trace else (summary, None)


def summarise(eps):
    c = np.array([e["cost"] for e in eps])
    return {"n": len(eps), "mean_cost": float(c.mean()), "median_cost": float(np.median(c)),
            "p75_cost": float(np.percentile(c, 75)), "runaway_rate": float(np.mean([e["runaway"] for e in eps])),
            "errors": int(sum(e["error"] is not None for e in eps))}


def evaluate(path, task="cstr", split="search", params=None, workers=2):
    check_source(path)
    base, n = SPLITS[split]
    jobs = [(os.path.abspath(path), task, base + i, params, False, False) for i in range(n)]
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(run_episode, jobs)
    else:
        res = [run_episode(j) for j in jobs]
    eps = [r[0] for r in res]
    return summarise(eps), eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controller")
    ap.add_argument("--task", default="cstr")
    ap.add_argument("--split", default="search", choices=sorted(SPLITS))
    ap.add_argument("--params", default=None, help="JSON dict passed to Controller(params)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    s, eps = evaluate(args.controller, args.task, args.split, json.loads(args.params) if args.params else None,
                      args.workers)
    print(f"{os.path.basename(args.controller)} task={args.task} split={args.split} n={s['n']}: mean cost "
          f"{s['mean_cost']:.3f}, median {s['median_cost']:.3f}, p75 {s['p75_cost']:.3f}, runaway "
          f"{100 * s['runaway_rate']:.0f}%, errors {s['errors']} [{time.time() - t0:.0f}s]")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": s, "episodes": eps}, f, indent=1)


if __name__ == "__main__":
    main()
