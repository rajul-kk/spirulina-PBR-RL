"""Counterfactual: replay fitted strains (fit2.txt) in mfsim under alternative parameter sets."""
import json, re, sys, numpy as np
from multiprocessing import Pool
import importlib.util
from mfsim import run, LUMP
res = {json.loads(l)["batch"]: json.loads(l) for l in open("../trials/results.jsonl")}
fits = []
for line in open("fit2.txt"):
    m = re.match(r"(\S+): mu (\S+) Topt (\S+) tau (\S+)", line)
    b = m.group(1); fits.append((b, res[b]["inoculum"], float(m.group(2)) / LUMP, float(m.group(3)), float(m.group(4)), res[b]["total_harvested_mg"]))
def job(a):
    ctrl, params, f = a
    spec = importlib.util.spec_from_file_location("c", ctrl); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    cls = type("C", (m.Controller,), {"__init__": lambda self, p=None: m.Controller.__init__(self, params)})
    return np.mean([run(cls, inoc=f[1], seed=s, strain={"mumax": f[2], "topt": f[3], "tau": f[4]})["total"] for s in range(2)])
if __name__ == "__main__":
    ctrl = sys.argv[1]; sets = [json.loads(s) for s in sys.argv[2:]]
    with Pool(4) as p:
        outs = [p.map(job, [(ctrl, ps, f) for f in fits]) for ps in sets]
    for i, f in enumerate(fits):
        print(f"{f[0]:12s} inoc {f[1]:5d} actual {f[5]:7.0f} | " + " ".join(f"{o[i]:7.0f}" for o in outs))
    print("mean ratio vs set0:", " ".join(f"{np.mean(np.array(o)/np.array(outs[0])):.3f}" for o in outs))
