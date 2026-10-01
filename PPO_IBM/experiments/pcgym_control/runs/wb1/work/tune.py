"""Offline tuning on my own plant model (no pilot budget). Common random scenarios, small process pool.
Usage: python tune.py <controller.py> <n> <seed0> '<json list of [name, params, oracle]>'"""
import json, sys
from multiprocessing import Pool
import numpy as np
import plantsim

def job(a):
    path, seed, params, oracle = a
    C = plantsim.load(path)
    c, ra, tmax, _ = plantsim.run(C(params), seed, oracle=oracle)
    return c, ra, tmax

if __name__ == "__main__":
    path, n, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    variants = json.loads(sys.argv[4])
    seeds = list(range(seed0, seed0 + n))
    base = None
    with Pool(5) as pool:
        for name, params, oracle in variants:
            res = pool.map(job, [(path, s, params, oracle) for s in seeds])
            c = np.array([r[0] for r in res])
            if base is None:
                base = c
            d = c - base
            print(f"{name:28s} mean={c.mean():.4f} med={np.median(c):.4f} p90={np.percentile(c,90):.3f} max={c.max():.3f} "
                  f"runaway={sum(r[1] for r in res)} tmax={max(r[2] for r in res):.1f}  d_vs_first={d.mean():+.4f}+-{d.std()/np.sqrt(n):.4f}", flush=True)
