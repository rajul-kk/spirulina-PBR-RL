"""Offline comparison of controller settings in my own simulator (same seeds for every config).
usage: python tune.py <controller.py> <n> <seed0> <oracle 0/1> '<json list of param dicts>'
"""
import json
import sys

import numpy as np

import sim

if __name__ == "__main__":
    path, n, seed0, oracle = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4]))
    configs = json.loads(sys.argv[5])
    base = None
    for cfg in configs:
        res = sim.evaluate(path, range(seed0, seed0 + n), cfg, oracle, procs=4)
        c = np.array([r["cost"] for r in res])
        s = sim.summarize(res)
        line = "%-40s mean %.4f med %.4f p90 %.3f max %.3f runaways %d Tmax %.1f" % (
            json.dumps(cfg), s["mean"], s["median"], s["p90"], s["max"], s["runaways"], s["t_max"])
        if base is None:
            base = c
        else:
            d = c - base
            line += "  | diff vs first %+.4f +/- %.4f" % (d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
        print(line, flush=True)
