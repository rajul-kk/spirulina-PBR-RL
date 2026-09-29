"""Compare controller versions against my model's prediction for the same inoculum (removes inoculum effect)."""
import json, sys, numpy as np
from sim import simulate
from opt3 import P3
res = [json.loads(l) for l in open("../trials/results.jsonl")]
cache = {}
def ref(n0):
    if n0 not in cache:
        cache[n0] = simulate(P3(od_sp=1.7, k_final=3, burst_P=24, burst_B=2), n0=n0, mu_max=0.036, dens_pen=0.08)[0]
    return cache[n0]
by = {}
for r in res:
    lab = r["batch"].split("_", 1)[1] if "_" in r["batch"] else "?"
    lab = lab.split("i")[0] if lab.startswith("v3i") else lab
    by.setdefault(lab, []).append((r["inoculum"], r["total_harvested_mg"], r["culture_lost"]))
for lab, v in by.items():
    rat = [h/ref(n) for n, h, _ in v]
    print(f"{lab:10s} n={len(v):3d} ratio-to-model mean {np.mean(rat):.3f} sd {np.std(rat):.3f} lost {sum(x[2] for x in v)}")
