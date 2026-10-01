"""My own model of the plant and of the batch scenarios, written from the equations in
pcgym/model_classes.py and the scenario description in tasks.py. It does NOT import pcgym or
tasks.py; it is used for design work that costs no pilot-plant batches.

usage: python sim.py <controller.py> [--n N] [--seed0 S] [--params '{"H":20}'] [--jobs J]
"""
import argparse
import importlib.util
import json
import math
import sys
import time

import numpy as np

K0, ER = 7.2e10, 8750.0
DH = 5e4 / (1000.0 * 0.239)
UA = 5e4 / (1000.0 * 0.239 * 100.0)
N, TSIM = 120, 26.0
DT = TSIM / N


def f(ca, T, Tc, Ti, Caf):
    r = K0 * math.exp(-ER / T) * ca
    return (Caf - ca) - r, (Ti - T) + DH * r + UA * (Tc - T)


def plant_step(ca, T, Tc, Ti, Caf, nsub=20):
    h = DT / nsub
    for _ in range(nsub):
        a1, b1 = f(ca, T, Tc, Ti, Caf)
        a2, b2 = f(ca + 0.5 * h * a1, T + 0.5 * h * b1, Tc, Ti, Caf)
        a3, b3 = f(ca + 0.5 * h * a2, T + 0.5 * h * b2, Tc, Ti, Caf)
        a4, b4 = f(ca + h * a3, T + h * b3, Tc, Ti, Caf)
        ca += h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        T += h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
    return ca, T


def steps(rng, n, lo, hi, n_changes, min_len=20):
    while True:
        cuts = np.sort(rng.randint(min_len, n - min_len, n_changes))
        if np.all(np.diff(np.concatenate([[0], cuts, [n]])) >= min_len):
            break
    levels = rng.uniform(lo, hi, n_changes + 1)
    return np.repeat(levels, np.diff(np.concatenate([[0], cuts, [n]])))


def scenario(seed):
    rng = np.random.RandomState(seed)
    return {"sp": steps(rng, N, 0.86, 0.90, 2),
            "Ti": steps(rng, N, 348.5, 351.5, rng.randint(1, 3)),
            "Caf": steps(rng, N, 0.98, 1.02, rng.randint(1, 3)),
            "x0": np.array([rng.uniform(0.85, 0.91), rng.uniform(320.0, 326.0)]),
            "noise_seed": int(rng.randint(1 << 30))}


def load(path):
    spec = importlib.util.spec_from_file_location("ctrl_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_batch(Controller, seed, params=None, trace=False):
    sc = scenario(seed)
    nrng = np.random.RandomState(sc["noise_seed"])
    ctrl = Controller(params)
    ca, T = float(sc["x0"][0]), float(sc["x0"][1])
    cost, tmax = 0.0, T
    rows = []
    sq = np.zeros(N)
    for k in range(N):
        obs = {"t_min": k * DT, "Ca": ca + nrng.normal(0, 0.002), "T": T + nrng.normal(0, 0.2),
               "Ca_sp": float(sc["sp"][k])}
        if hasattr(ctrl, "truth"):
            ctrl.truth = (ca, T, float(sc["Ti"][k]), float(sc["Caf"][k]))
        u = float(ctrl.act(obs))
        u = min(max(u, 295.0), 302.0)
        ca, T = plant_step(ca, T, u, float(sc["Ti"][k]), float(sc["Caf"][k]))
        sq[k] = ((ca - sc["sp"][k]) / 0.01) ** 2
        tmax = max(tmax, T)
        if trace:
            xe = getattr(ctrl, "x", None)
            rows.append((k, obs["Ca"], obs["T"], obs["Ca_sp"], u, ca, T, sc["Ti"][k], sc["Caf"][k],
                         *(xe if xe is not None else (0, 0, 0, 0))))
    out = {"seed": seed, "cost": float(sq.mean()), "runaway": bool(tmax > 335.0), "t_max": tmax}
    if trace:
        out["rows"] = rows
    out["sq"] = sq
    return out


def _work(a):
    path, seed, params = a
    return run_batch(load(path).Controller, seed, params)


def evaluate(path, seeds, params=None, jobs=1):
    args = [(path, s, params) for s in seeds]
    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            res = pool.map(_work, args)
    else:
        res = [_work(a) for a in args]
    return res


def summarise(res):
    c = np.array([r["cost"] for r in res])
    return {"n": len(c), "mean": float(c.mean()), "median": float(np.median(c)),
            "p90": float(np.percentile(c, 90)), "max": float(c.max()),
            "runaways": int(sum(r["runaway"] for r in res)),
            "tmax": float(max(r["t_max"] for r in res))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ctrl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    res = evaluate(a.ctrl, range(a.seed0, a.seed0 + a.n), json.loads(a.params), a.jobs)
    s = summarise(res)
    s["sec"] = round(time.time() - t0, 1)
    print(json.dumps(s))
