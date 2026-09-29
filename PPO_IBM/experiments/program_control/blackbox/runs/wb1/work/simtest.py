"""Run a controller file against my mean-field model (synthetic sensors) for quick checks."""
import sys, importlib.util, numpy as np
from mfmodel import Model
def load(path):
    spec = importlib.util.spec_from_file_location("c", path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m.Controller
def run(C, cells, mumax=0.04, seed=0, params=None, trace=False):
    m = Model(cells=cells, mumax=mumax, seed=seed); c = C(params); hs = []
    for t in range(7200):
        o = m.obs(); o["t"] = t
        s, I, f = c.act(o)
        if trace and t % 600 == 300: print(f"  t={t*0.02:5.1f}h od={m.od:.3f} est={c.od:.3f} c={m.c:.2f}/{c.c:.2f} I={I:.0f}")
        h = m.step(s, I, f)
        if h: hs.append(round(h))
        if m.cells < 10: return m.harvested, hs, True
    return m.harvested, hs, False
if __name__ == "__main__":
    C = load(sys.argv[1])
    for cells in [40, 250, 3000]:
        print(cells, run(C, cells, trace=True))
