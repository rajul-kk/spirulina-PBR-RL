"""Open-loop harvest schedule optimisation on mfsim (nominal strain) per inoculum."""
import sys, json, numpy as np
from multiprocessing import Pool
import ctrl_v1 as cv
from mfsim import run

def make(sched, params):
    class C(cv.Controller):
        def __init__(self, p=None):
            super().__init__(params)
        def act(self, obs):
            s, l, _ = super().act(obs)
            k = int(obs["t"]) // 600
            return s, l, (sched[k] if k < 11 else 0.0)
    return C

def val(a):
    sched, inoc, params = a
    return np.mean([run(make(sched, params), inoc=inoc, seed=s, strain={"mumax": 0.04, "topt": 36, "tau": 2.5})["total"] for s in range(2)])

if __name__ == "__main__":
    inoc = int(sys.argv[1]); params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    sched = [0.0] * 8 + [0.5] * 3
    levels = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    with Pool(4) as pool:
        best = val((sched, inoc, params))
        for it in range(3):
            improved = False
            for k in range(11):
                cands = [sched[:k] + [v] + sched[k + 1:] for v in levels if v != sched[k]]
                vals = pool.map(val, [(c, inoc, params) for c in cands])
                j = int(np.argmax(vals))
                if vals[j] > best + 1:
                    best, sched, improved = vals[j], cands[j], True
            print(inoc, it, round(best), sched, flush=True)
            if not improved:
                break
