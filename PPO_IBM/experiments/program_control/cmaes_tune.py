"""Classical baseline: CMA-ES over the sensor expert's knobs, scored on the search split only.

The question every LLM-written controller has to answer is "better than simply tuning the
existing law?", so this is the bar. Standard CMA-ES (Hansen's tutorial, rank-mu + rank-one
updates) in a [0, 1]-normalised box; no external dependency.

  python experiments/program_control/cmaes_tune.py --gens 12 --popsize 8 --workers 2
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

CTRL = os.path.join(HERE, "controllers", "sensor_expert.py")
BOUNDS = {"stir": (50.0, 200.0), "light": (0.0, 2000.0), "setpoint": (0.2, 1.5),
          "gain": (0.1, 5.0), "cap": (0.05, 0.5), "turb_per_od": (150.0, 350.0)}
KEYS = list(BOUNDS)
START = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30, "turb_per_od": 250.0}


def decode(z):
    z = np.clip(z, 0.0, 1.0)
    return {k: float(lo + zi * (hi - lo)) for zi, (k, (lo, hi)) in zip(z, BOUNDS.items())}


def encode(p):
    return np.array([(p[k] - lo) / (hi - lo) for k, (lo, hi) in BOUNDS.items()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "cmaes"))
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
    best = (-np.inf, None, None)

    # Generation 0 also scores the hand-tuned start, so the table has its search-split number.
    s0, _ = evaluate(CTRL, "search", 2, decode(mean), workers=args.workers)
    best = (s0["fitness"], decode(mean), s0)
    with open(log_path, "a") as f:
        f.write(json.dumps({"gen": 0, "params": decode(mean), "summary": s0, "start": True}) + "\n")
    print(f"start: fitness {s0['fitness']:.0f} median {s0['median_mg']:.0f} crash {s0['crash_rate']:.0%}", flush=True)

    for g in range(1, args.gens + 1):
        t0 = time.time()
        vals, vecs = np.linalg.eigh(C)
        B, D = vecs, np.sqrt(np.maximum(vals, 1e-20))
        zs = rng.standard_normal((lam, n))
        ys = zs @ np.diag(D) @ B.T
        xs = mean + sigma * ys
        scored = []
        for x in xs:
            p = decode(x)
            s, _ = evaluate(CTRL, "search", 2, p, workers=args.workers)
            # Out-of-box candidates are clipped for evaluation and lightly penalised so the
            # distribution doesn't drift into the walls.
            fit = s["fitness"] - 2000.0 * float(np.sum(np.clip(-x, 0, None) + np.clip(x - 1, 0, None)))
            scored.append((fit, x, p, s))
            with open(log_path, "a") as f:
                f.write(json.dumps({"gen": g, "params": p, "summary": s}) + "\n")
            if s["fitness"] > best[0]:
                best = (s["fitness"], p, s)
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
        top = scored[0]
        print(f"gen {g:2d}: best-of-gen {top[3]['fitness']:.0f} (median {top[3]['median_mg']:.0f}, "
              f"crash {top[3]['crash_rate']:.0%}) | best-so-far {best[0]:.0f} | sigma {sigma:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        with open(os.path.join(args.out, "best.json"), "w") as f:
            json.dump({"fitness": best[0], "params": best[1], "summary": best[2]}, f, indent=1)


if __name__ == "__main__":
    main()
