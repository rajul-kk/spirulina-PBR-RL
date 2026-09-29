"""Supervisory controller v1 for the Spirulina PBR (run bb2).
- stir fixed (stir changes the turbidity reading, not growth).
- biomass estimate from filtered turbidity via the fitted calibration
  NTU = A*(1-exp(-X/D))*(1-C*hour/144).
- light scheduled on estimated density (moderate at low X, full at high X).
- harvest: hold post-harvest density at X_HOLD until the end game, then dump.
"""
import math
from collections import deque

P = dict(
    STIR=100.0,
    A=700.0, D=780.0, C=0.14,
    L_LO=1000.0, L_HI=2000.0, X_LO=150.0, X_HI=600.0,
    X_HOLD=850.0,        # post-harvest density target before end game
    END_K=8,             # harvest k (at 12k h): k==END_K -> X_POST_END; k>END_K -> F_END
    X_POST_END=500.0,     # post-harvest target at the first end-game harvest
    F_END=0.5,
    X_MIN_POST=40.0,     # never harvest below this estimated post-harvest density
    X_CAP=3000.0,
)

class Controller:
    STEPS_PER_INT = 600  # 12 h / 0.02 h

    def __init__(self, params=None):
        self.p = dict(P)
        if params:
            self.p.update(params)
        self.k = 0
        self.ntu = None
        self.req_sum = 0.0
        self.req_n = 0

    def _xest(self, ntu, hour):
        p = self.p
        s = ntu / (p['A'] * max(0.5, 1.0 - p['C'] * hour / 144.0))
        s = min(s, 0.985)
        return min(p['X_CAP'], -p['D'] * math.log(1.0 - max(s, 0.0)))

    def act(self, obs):
        p = self.p
        t = self.k
        self.k += 1
        hour = t * 0.02
        z = float(obs.get('turbidity_ntu', 0.0))
        if self.ntu is None:
            self.ntu = z
        else:
            self.ntu += 0.02 * (z - self.ntu)   # ~1 h time constant
        X = self._xest(self.ntu, hour)

        # light schedule
        if X <= p['X_LO']:
            L = p['L_LO']
        elif X >= p['X_HI']:
            L = p['L_HI']
        else:
            L = p['L_LO'] + (p['L_HI'] - p['L_LO']) * (X - p['X_LO']) / (p['X_HI'] - p['X_LO'])

        # harvest: position in current 12 h interval
        pos = t % self.STEPS_PER_INT
        if pos == 0:
            self.req_sum = 0.0
            self.req_n = 0
        k_next = t // self.STEPS_PER_INT + 1         # index of the upcoming harvest (12h*k)
        remaining_h = (self.STEPS_PER_INT - pos) * 0.02
        # predicted density at the harvest instant (crude growth guess 1%/h)
        Xp = X * math.exp(0.01 * remaining_h)
        if k_next >= p['END_K'] + 1:
            f = p['F_END']
        elif k_next == p['END_K']:
            f = 1.0 - p['X_POST_END'] / max(Xp, 1e-6)
        else:
            f = 1.0 - p['X_HOLD'] / max(Xp, 1e-6)
        f = max(0.0, min(0.5, f))
        # safety floor
        if Xp * (1.0 - f) < p['X_MIN_POST']:
            f = max(0.0, 1.0 - p['X_MIN_POST'] / max(Xp, 1e-6))
        # make the interval mean equal f given what was already requested
        n_rem = self.STEPS_PER_INT - pos
        r = (f * self.STEPS_PER_INT - self.req_sum) / n_rem
        r = max(0.0, min(0.5, r))
        self.req_sum += r
        self.req_n += 1
        return (p['STIR'], L, r)
