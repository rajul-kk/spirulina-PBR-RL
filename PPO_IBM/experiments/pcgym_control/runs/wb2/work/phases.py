"""Own-model study: where does the cost come from? Splits each simulated batch's cost into phases:
start-up (first 10 samples), 10 samples after a setpoint change, 10 samples after a feed change
(Ti or Caf), and the quiet remainder. Contributions are in batch-cost units (sum / 120).

usage: python phases.py <controller.py> --n N --params '{...}' ['{...}' ...]
"""
import argparse
import json
import multiprocessing as mp

import numpy as np

import sim


def phase_masks(sc, w=10):
    n = sim.N
    lab = np.zeros(n, dtype=int)          # 0 quiet
    for name, code in (("Ti", 3), ("Caf", 3), ("sp", 2)):
        for c in np.nonzero(np.diff(sc[name]))[0] + 1:
            lab[c:c + w] = code
    lab[:w] = 1
    return lab


def work(a):
    path, seed, params = a
    r = sim.run_batch(sim.load(path).Controller, seed, params)
    lab = phase_masks(sim.scenario(seed))
    return [r["cost"]] + [float(r["sq"][lab == c].sum() / sim.N) for c in (1, 2, 3, 0)] + [r["t_max"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ctrl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--params", nargs="+", default=["{}"])
    a = ap.parse_args()
    with mp.Pool(a.jobs) as pool:
        for ps in a.params:
            res = np.array(pool.map(work, [(a.ctrl, s, json.loads(ps)) for s in range(a.seed0, a.seed0 + a.n)]))
            m = res.mean(axis=0)
            print("%-70s mean %.4f med %.4f | start %.4f sp %.4f feed %.4f quiet %.4f | Tmax %.1f" % (
                ps, m[0], np.median(res[:, 0]), m[1], m[2], m[3], m[4], res[:, 5].max()), flush=True)
