"""My own model of the plant, written from the equations in pcgym/model_classes.py (cstr_ode) and
the scenario description in tasks.py. It does NOT import tasks.py or pcgym: no pilot budget is used.

Usage: python plantsim.py <controller.py> [--n N] [--seed0 S] [--oracle] [--params '{"k": v}']
"""
import argparse
import importlib.util
import json
import math
import sys
import time

import numpy as np

K0, EA = 7.2e10, 8750.0
DH = 5e4 / (1000 * 0.239)           # (-deltaHr)/(rho C)
UA = 5e4 / (1000 * 0.239 * 100)     # UA/(rho C V)
N, TSIM = 120, 26.0
DT = TSIM / N
A_LOW, A_HIGH = 295.0, 302.0
NOISE_CA, NOISE_T = 0.002, 0.2
T_RUNAWAY, ERR_SCALE = 335.0, 0.01


def rhs(ca, T, tc, ti, caf):
    r = K0 * math.exp(-EA / T) * ca
    return caf - ca - r, ti - T + DH * r + UA * (tc - T)


def step_true(ca, T, tc, ti, caf, nsub=20):
    h = DT / nsub
    for _ in range(nsub):
        a1, b1 = rhs(ca, T, tc, ti, caf)
        a2, b2 = rhs(ca + 0.5 * h * a1, T + 0.5 * h * b1, tc, ti, caf)
        a3, b3 = rhs(ca + 0.5 * h * a2, T + 0.5 * h * b2, tc, ti, caf)
        a4, b4 = rhs(ca + h * a3, T + h * b3, tc, ti, caf)
        ca += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        T += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
    return ca, T


def _steps(rng, n, lo, hi, n_changes, min_len=20):
    while True:
        cuts = np.sort(rng.randint(min_len, n - min_len, n_changes))
        if np.all(np.diff(np.concatenate([[0], cuts, [n]])) >= min_len):
            break
    levels = rng.uniform(lo, hi, n_changes + 1)
    return np.repeat(levels, np.diff(np.concatenate([[0], cuts, [n]])))


def scenario(seed):
    rng = np.random.RandomState(seed)
    return {"sp": _steps(rng, N, 0.86, 0.90, 2),
            "Ti": _steps(rng, N, 348.5, 351.5, rng.randint(1, 3)),
            "Caf": _steps(rng, N, 0.98, 1.02, rng.randint(1, 3)),
            "x0": np.array([rng.uniform(0.85, 0.91), rng.uniform(320.0, 326.0)]),
            "noise_seed": int(rng.randint(1 << 30))}


def run(ctrl, seed, oracle=False, trace=False):
    sc = scenario(seed)
    nrng = np.random.RandomState(sc["noise_seed"])
    ca, T = float(sc["x0"][0]), float(sc["x0"][1])
    cost, tmax, rows = 0.0, T, []
    for k in range(N):
        obs = {"t_min": k * TSIM / N, "Ca": ca + nrng.normal(0, NOISE_CA),
               "T": T + nrng.normal(0, NOISE_T), "Ca_sp": float(sc["sp"][k])}
        if oracle:
            obs["_truth"] = (ca, T, float(sc["Ti"][k]), float(sc["Caf"][k]))
        u = float(np.clip(ctrl.act(obs), A_LOW, A_HIGH))
        ca, T = step_true(ca, T, u, float(sc["Ti"][k]), float(sc["Caf"][k]))
        sq = ((ca - sc["sp"][k]) / ERR_SCALE) ** 2
        cost += sq
        tmax = max(tmax, T)
        if trace:
            rows.append((k, obs["Ca"], obs["T"], obs["Ca_sp"], u, ca, T, sc["Ti"][k], sc["Caf"][k], sq))
    if getattr(ctrl, "n_fallback", 0):
        print("WARNING: controller fallback used", ctrl.n_fallback, "times, seed", seed, flush=True)
    return cost / N, tmax > T_RUNAWAY, tmax, rows


def load(path):
    spec = importlib.util.spec_from_file_location("ctrl_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Controller


def evaluate(path, seeds, params=None, oracle=False):
    C = load(path)
    out = []
    for s in seeds:
        c, ra, tmax, _ = run(C(params), s, oracle=oracle)
        out.append((s, c, ra, tmax))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ctrl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--params", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    res = evaluate(a.ctrl, range(a.seed0, a.seed0 + a.n), json.loads(a.params) if a.params else None, a.oracle)
    c = np.array([r[1] for r in res])
    if a.verbose:
        for r in res:
            print(r)
    print(f"n={len(c)} mean={c.mean():.4f} median={np.median(c):.4f} p90={np.percentile(c, 90):.4f} "
          f"max={c.max():.3f} runaways={sum(r[2] for r in res)} tmax={max(r[3] for r in res):.2f} "
          f"time={time.time() - t0:.1f}s")
