"""Classical baseline on the PC-Gym task: CMA-ES over the gains of controllers/pid.py, scored
on the search split only. Same algorithm and budget as program_control/cmaes_tune.py
(12 generations x 8 candidates x 12 search episodes, plus the start point).

  python cmaes_tune.py --seed 0 --out results/cmaes/s0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import evaluate  # noqa: E402

CTRL = os.path.join(HERE, "controllers", "pid.py")
BOUNDS = {"kp": (0.0, 300.0), "ki": (0.0, 1000.0), "kd": (0.0, 2.0), "bias": (295.0, 302.0)}
KEYS = list(BOUNDS)
START = {"kp": 30.0, "ki": 60.0, "kd": 0.0, "bias": 299.0}


def decode(z):
    z = np.clip(z, 0.0, 1.0)
    return {k: float(lo + zi * (hi - lo)) for zi, (k, (lo, hi)) in zip(z, BOUNDS.items())}


def encode(p):
    return np.array([(p[k] - lo) / (hi - lo) for k, (lo, hi) in BOUNDS.items()])


def fitness(s):
    return -(s["mean_cost"] + 10.0 * s["runaway_rate"])     # maximised


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cstr")
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n, lam = len(KEYS), args.popsize
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    mean, sigma = encode(START), args.sigma
    C, pc, ps = np.eye(n), np.zeros(n), np.zeros(n)
    log_path = os.path.join(args.out, "log.jsonl")

    s0, _ = evaluate(CTRL, args.task, "search", decode(mean), workers=args.workers)
    best = (fitness(s0), decode(mean), s0)
    with open(log_path, "a") as f:
        f.write(json.dumps({"gen": 0, "params": decode(mean), "summary": s0, "start": True}) + "\n")
    print(f"start: mean cost {s0['mean_cost']:.3f} runaway {s0['runaway_rate']:.0%}", flush=True)

    for g in range(1, args.gens + 1):
        t0 = time.time()
        vals, vecs = np.linalg.eigh(C)
        B, D = vecs, np.sqrt(np.maximum(vals, 1e-20))
        xs = mean + sigma * (rng.standard_normal((lam, n)) @ np.diag(D) @ B.T)
        scored = []
        for x in xs:
            p = decode(x)
            s, _ = evaluate(CTRL, args.task, "search", p, workers=args.workers)
            # Out-of-box candidates are clipped for evaluation and lightly penalised.
            fit = fitness(s) - 5.0 * float(np.sum(np.clip(-x, 0, None) + np.clip(x - 1, 0, None)))
            scored.append((fit, x, p, s))
            with open(log_path, "a") as f:
                f.write(json.dumps({"gen": g, "params": p, "summary": s}) + "\n")
            if fitness(s) > best[0]:
                best = (fitness(s), p, s)
        scored.sort(key=lambda r: -r[0])
        x_sel = np.array([r[1] for r in scored[:mu]])
        y_sel = (x_sel - mean) / sigma
        old_mean = mean
        mean = w @ x_sel
        y_w = (mean - old_mean) / sigma
        inv_sqrt_C = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ y_w)
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * g)) / chi_n < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (y_sel.T * w) @ y_sel)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
        print(f"gen {g:2d}: best-of-gen cost {scored[0][3]['mean_cost']:.3f} | best-so-far "
              f"{-best[0]:.3f} | sigma {sigma:.3f} [{time.time() - t0:.0f}s]", flush=True)
        with open(os.path.join(args.out, "best.json"), "w") as f:
            json.dump({"fitness": best[0], "params": best[1], "summary": best[2]}, f, indent=1)
    with open(os.path.join(args.out, "DONE"), "w") as f:
        f.write("12 generations complete\n")


if __name__ == "__main__":
    main()
