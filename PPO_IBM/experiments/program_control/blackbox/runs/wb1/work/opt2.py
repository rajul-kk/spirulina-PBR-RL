"""Oracle-state policy search on my mean-field model (true OD known to the policy)."""
import numpy as np, sys, itertools
from multiprocessing import Pool
from mfmodel import Model
def sim(args):
    cells, mumax, th = args
    m = Model(cells=cells, mumax=mumax, seed=1)
    I = 300.0
    for t in range(7200):
        od = m.od
        rpm = th["r_lo"] if od < th["od_r"] else th["r_hi"]
        It = min(th["imax"], th["ia"] + th["ib"] * od)
        I = min(It, I + th["ramp"] * 0.02) if It > I else It
        k = t // 600 + 1
        if k >= th["bleed"]: f = 0.5
        else: f = min(0.5, max(0.0, 1 - th["odt"] / od)) if od > th["odt"] else 0.0
        m.step(rpm, I, f)
    return m.harvested
BASE = dict(r_lo=70, r_hi=70, od_r=99, imax=1800, ia=450, ib=1400, ramp=200, odt=2.0, bleed=9)
INOC = [40, 120, 250, 400, 1000, 3000]
MUS = [0.032, 0.042]
def score(th, pool):
    jobs = [(c, mu, th) for c in INOC for mu in MUS]
    r = pool.map(sim, jobs)
    return np.array(r).reshape(len(INOC), len(MUS)).mean(axis=1)
if __name__ == "__main__":
    grid = {"odt": [1.0, 1.5, 2.0, 3.0, 4.0, 6.0], "bleed": [7, 8, 9, 10, 11], "r_hi": [60, 70, 80, 90, 100, 120],
            "r_lo": [50, 60, 70, 80], "od_r": [0.5, 1.0, 2.0, 99], "imax": [1500, 1650, 1800, 2000],
            "ib": [800, 1400, 2500], "ia": [300, 450, 600], "ramp": [50, 100, 200, 1000]}
    th = dict(BASE)
    with Pool(6) as pool:
        base = score(th, pool); print("base", base.round(0), flush=True)
        for sweep in range(2):
            for key, vals in grid.items():
                res = []
                for v in vals:
                    t2 = dict(th); t2[key] = v
                    s = score(t2, pool); res.append((np.log(s).sum(), v, s))
                best = max(res, key=lambda x: x[0])
                th[key] = best[1]
                print(key, [(v, round(float(np.log(s).sum()), 3)) for _, v, s in res], "->", best[1], best[2].round(0), flush=True)
        print(th)
