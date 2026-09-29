import numpy as np, sys
from mfmodel import Model
def sim(cells, rpm=70, odt=3.0, Imax=1800, bleed=9, ramp=200.0, mumax=0.04, seed=0, verbose=False):
    m = Model(cells=cells, mumax=mumax, seed=seed)
    I = 300.0; hs=[]
    for t in range(7200):
        od = m.od
        It = min(Imax, 450+1400*od)
        I = min(It, I + ramp*0.02) if It > I else It
        k = t//600 + 1  # upcoming event index (1..11)
        if k >= bleed: f = 0.5
        else:
            # frac so that post-harvest od = odt, using current od projected
            f = min(0.5, max(0.0, 1 - odt/od)) if od > odt else 0.0
        h = m.step(rpm, I, f)
        if h: hs.append(round(h))
        if verbose and t % 600 == 599: print(t, round(od,3), round(I), m.last)
    return m.harvested, hs
if __name__ == "__main__":
    for cells in [30, 100, 250, 400, 1000, 5000]:
        best = None
        for rpm in [60, 70, 80, 90]:
            for odt in [1.0, 2.0, 3.0, 4.0, 6.0]:
                for bleed in [7, 8, 9, 10, 11, 12]:
                    H, hs = sim(cells, rpm, odt, 1800, bleed)
                    if best is None or H > best[0]: best = (H, rpm, odt, bleed, hs)
        print(cells, best)
