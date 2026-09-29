import sys, importlib.util, numpy as np, json
from multiprocessing import Pool
from mfsim import run
INOCS = [30, 60, 100, 150, 200, 300, 400, 700, 1500, 4000]
CTRL = "ctrl_v1.py"

def _load(path):
    spec = importlib.util.spec_from_file_location("c", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def _job(a):
    path, params, inoc, seed = a
    m = _load(path)
    cls = type("C", (m.Controller,), {"__init__": lambda self, p=None: m.Controller.__init__(self, params)})
    r = run(cls, inoc=inoc, seed=seed)
    return inoc, r["total"], r["lost"]

def evaluate(params, seeds=range(4), inocs=INOCS, ctrl=None, pool=None):
    jobs = [(ctrl or CTRL, params, i, s * 1000 + i) for i in inocs for s in seeds]
    res = (pool.map(_job, jobs) if pool else list(map(_job, jobs)))
    out = {}
    for inoc, tot, lost in res:
        out.setdefault(inoc, []).append({"total": tot, "lost": lost})
    return out

def summary(out):
    # weight inocula like the plant distribution: 10% 30-80, 70% 100-400, 20% 600-5000
    w = {30: .05, 60: .05, 100: .14, 150: .14, 200: .14, 300: .14, 400: .14, 700: .07, 1500: .07, 4000: .06}
    tot = [r["total"] for v in out.values() for r in v]
    wmean = sum(w.get(k, 0) * np.mean([r["total"] for r in v]) for k, v in out.items())
    lost = sum(r["lost"] for v in out.values() for r in v)
    by = " ".join(f"{k}:{np.median([r['total'] for r in v])/1000:.1f}" for k, v in out.items())
    return wmean, np.percentile(tot, 25), lost, by
