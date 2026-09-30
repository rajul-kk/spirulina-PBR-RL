import numpy as np, sys, itertools
from mfsim import run
LP = [(0.06,600),(0.2,750),(0.4,1050),(0.6,1400),(0.8,1700),(1.2,1800)]
def light(od):
    return float(np.interp(od,[p[0] for p in LP],[p[1] for p in LP]))
def mkpol(caps, stir=60, burst=None):
    # caps[k] = od cap after event k+1 (k=0..10); h = clip(1-cap/od_pred,0,0.5) decided from od at time t
    def pol(t, od, c):
        k = min(int(t // 12), 10)
        h = min(0.5, max(0.0, 1 - caps[k] / od))
        s = stir
        if burst and (t % burst[0]) < burst[1] and t > 24: s = burst[2]
        return s, light(od), h
    return pol
SCEN = [(0.07,.1),(0.2,.2),(0.3,.25),(0.5,.25),(0.8,.08),(2.0,.07),(5.0,.05)]
MUS = (0.024, 0.031, 0.036)
def score(caps, **kw):
    tot = 0
    for od0, w in SCEN:
        for mu in MUS: tot += w * run(mkpol(caps, **kw), od0, mu_eff=mu, dt=0.2) / len(MUS)
    return tot
def optimize(caps, **kw):
    caps = list(caps); best = score(caps, **kw)
    for it in range(3):
        for k in range(11):
            for f in (0.7, 0.85, 1.15, 1.4):
                trial = list(caps); trial[k] = caps[k]*f
                s = score(trial, **kw)
                if s > best: best, caps = s, trial
        print(it, round(best), [round(c,2) for c in caps], flush=True)
    return caps, best
if __name__ == '__main__':
    init = [4.0]*7 + [2.0, 1.0, 0.5, 0.2]
    for kw in (dict(stir=60), dict(stir=80), dict(stir=60, burst=(24, 2, 200))):
        print(kw); optimize(init, **kw)
