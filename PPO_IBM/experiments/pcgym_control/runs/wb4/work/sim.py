"""My own model of the plant and of the scenario distribution, written from the equations in
pcgym/model_classes.py and the scenario description in tasks.py. Does NOT import pcgym or tasks.
Used for offline design only; the real plant is only reached through plant_trial.py.

usage: python sim.py <controller.py> [--n 100] [--seed0 1000] [--oracle] [--params '{"H":10}'] [--procs 4]
"""
import argparse
import importlib.util
import json
import math
import sys
import time

import numpy as np

K0, EA = 7.2e10, 8750.0
DH = 5e4 / 239.0
UA = 5e4 / 23900.0
N, TSIM = 120, 26.0
DT = TSIM / N


def rhs(ca, T, tc, ti, caf):
    r = K0 * math.exp(-EA / T) * ca
    return caf - ca - r, ti - T + DH * r + UA * (tc - T)


def plant_step(ca, T, tc, ti, caf, nsub=40):
    h = DT / nsub
    for _ in range(nsub):
        a1, b1 = rhs(ca, T, tc, ti, caf)
        a2, b2 = rhs(ca + 0.5 * h * a1, T + 0.5 * h * b1, tc, ti, caf)
        a3, b3 = rhs(ca + 0.5 * h * a2, T + 0.5 * h * b2, tc, ti, caf)
        a4, b4 = rhs(ca + h * a3, T + h * b3, tc, ti, caf)
        ca += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        T += h * (b1 + 2 * b2 + 2 * b3 + b4) / 6
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


def load_controller(path):
    spec = importlib.util.spec_from_file_location("ctrl_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Controller


def run_batch(Ctrl, seed, params=None, oracle=False, trace=False):
    sc = scenario(seed)
    nrng = np.random.RandomState(sc["noise_seed"])
    ca, T = float(sc["x0"][0]), float(sc["x0"][1])
    c = Ctrl(params)
    if oracle:
        truth = {}
        c._estimate = lambda obs: np.array([truth["ca"], truth["T"], truth["ti"], truth["caf"]])
    cost, tmax = 0.0, T
    rows = []
    sq = np.zeros(N)
    for k in range(N):
        obs = {"t_min": k * DT, "Ca": ca + nrng.normal(0, 0.002), "T": T + nrng.normal(0, 0.2),
               "Ca_sp": float(sc["sp"][k])}
        if oracle:
            truth.update(ca=ca, T=T, ti=float(sc["Ti"][k]), caf=float(sc["Caf"][k]))
        u = float(np.clip(c.act(dict(obs)), 295.0, 302.0))
        ca0, T0 = ca, T
        ca, T = plant_step(ca, T, u, float(sc["Ti"][k]), float(sc["Caf"][k]))
        sq[k] = ((ca - sc["sp"][k]) / 0.01) ** 2
        tmax = max(tmax, T)
        if trace:
            est = getattr(c, "est", None)
            rows.append([k, obs["Ca"], obs["T"], obs["Ca_sp"], u, ca, T, sc["Ti"][k], sc["Caf"][k],
                         ca0, T0] + (list(est) if est is not None else [np.nan] * 4))
    out = {"seed": seed, "cost": float(sq.mean()), "runaway": bool(tmax > 335.0), "t_max": tmax}
    if trace:
        out["rows"] = np.array(rows)
        out["sq"] = sq
    return out


def _job(args):
    path, seed, params, oracle = args
    return run_batch(load_controller(path), seed, params, oracle)


def evaluate(path, seeds, params=None, oracle=False, procs=4):
    jobs = [(path, s, params, oracle) for s in seeds]
    if procs > 1:
        import multiprocessing as mp
        with mp.Pool(procs) as pool:
            res = pool.map(_job, jobs)
    else:
        res = [_job(j) for j in jobs]
    return res


def summarize(res):
    c = np.array([r["cost"] for r in res])
    return {"n": len(c), "mean": float(c.mean()), "median": float(np.median(c)),
            "p90": float(np.percentile(c, 90)), "max": float(c.max()),
            "sem": float(c.std(ddof=1) / math.sqrt(len(c))) if len(c) > 1 else 0.0,
            "runaways": int(sum(r["runaway"] for r in res)),
            "t_max": float(max(r["t_max"] for r in res))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("controller")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--params", default=None)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    params = json.loads(a.params) if a.params else None
    t0 = time.time()
    res = evaluate(a.controller, range(a.seed0, a.seed0 + a.n), params, a.oracle, a.procs)
    s = summarize(res)
    s["sec"] = round(time.time() - t0, 1)
    print(json.dumps(s))
