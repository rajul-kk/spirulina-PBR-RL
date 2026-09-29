"""Supervisory controller, Spirulina PBR (run bb1).

Design (from pilot batches, see LAB_NOTEBOOK.md):
- Stir at the 50 rpm minimum: shear/heat from stirring cut growth (50 rpm gave ~+40 % yield over 120).
- Light scheduled on estimated biomass: ~900 umol at low density (light already saturating; less
  heat), rising to 1500 above ~400 mg/L (self-shading); 2000 overheats the tank (thermostat
  saturates, +3-5 C) without extra growth.  A slow integral servo lowers the ceiling if the
  broth runs above 37.5 C.
- Harvest: productivity (mg/L/h) keeps rising with density up to ~800 mg/L and collapses above
  ~2000, and anything left in the tank after the last (132 h) harvest is lost.  A dynamic
  programme on a growth model fitted to the lab assays gives: never harvest small cultures before
  96 h; for large ones harvest down to ~650 mg/L each 12 h; at 96 h cut to ~420 mg/L; then take
  the 0.5 maximum at 108/120/132 h.
- Biomass is estimated from turbidity with a calibration fitted on the assays (turbidity reads
  low at high density and as filaments clump over days).  For safety the estimate used for
  harvesting is bounded by what the culture could have grown to from its first reading, so a
  faulty/high turbidity signal cannot drain a small culture.
"""
import math

# ln NTU = c0 + c1*l + c2*t + c3*l^2 + c4*t*l,  l = ln DW (mg/L), t = hour  (stir 50 rpm)
C = (-2.22876449, 1.86843690, 1.61463661e-03, -9.31409675e-02, -7.22231493e-04)
P = dict(stir=50.0, L_lo=900.0, L_hi=1500.0, X_lo=150.0, X_hi=400.0, T_cap=37.5, L_min=800.0,
         X_hold=650.0, X_96=420.0, X_108=150.0, mu_bound=0.035, bound_margin=1.3)


def ntu_to_dw(n, t):
    c0, c1, c2, c3, c4 = C
    n = max(n, 1.0)
    a = c3; b = c1 + c4 * t; c = c0 + c2 * t - math.log(n)
    lv = -b / (2 * a)                      # turbidity saturates beyond the vertex
    disc = b * b - 4 * a * c
    if disc <= 0:
        return math.exp(lv)
    return math.exp(min((-b + math.sqrt(disc)) / (2 * a), lv))


class Controller:
    def __init__(self, params=None):
        self.p = dict(P)
        if params:
            self.p.update(params)
        self.ntu = None
        self.T = None
        self.Lcap = 2000.0
        self.bound = None          # upper bound on plausible biomass (mg/L)
        self.win = -1              # index of current 12 h harvest window
        self.req_sum = 0.0
        self.req_n = 0
        self.last_hr = 0.0

    def act(self, obs):
        p = self.p
        hr = float(obs.get('t', 0)) * 0.02
        n = obs.get('turbidity_ntu', None)
        T = obs.get('temp_c', None)
        if n is None or not math.isfinite(n):
            n = self.ntu if self.ntu is not None else 50.0
        if T is None or not math.isfinite(T) or T < 15.0 or T > 50.0:
            T = self.T if self.T is not None else 35.0

        # window bookkeeping: at each 12 h mark the pump applied the mean request of the window
        w = int(math.floor(hr / 12.0 + 1e-9))
        if w != self.win:
            if self.win >= 0 and self.req_n > 0 and self.bound is not None:
                self.bound *= (1.0 - self.req_sum / self.req_n)
            self.win = w
            self.req_sum = 0.0
            self.req_n = 0
            if self.ntu is not None and n < 0.8 * self.ntu:
                self.ntu = n                  # jump after a harvest: re-seed the filter

        # filters
        if self.ntu is None:
            self.ntu = n
        else:
            self.ntu += 0.03 * (n - self.ntu)     # ~0.7 h time constant
        self.T = T if self.T is None else self.T + 0.02 * (T - self.T)
        X = ntu_to_dw(self.ntu, hr)

        # plausibility bound on biomass
        dt = max(0.0, hr - self.last_hr)
        self.last_hr = hr
        if self.bound is None:
            if hr >= 0.5 or self.req_n >= 25:
                self.bound = X * p['bound_margin']
        else:
            self.bound *= math.exp(p['mu_bound'] * dt)
        Xh = X if self.bound is None else min(X, self.bound)

        # light
        self.Lcap = min(2000.0, max(p['L_min'], self.Lcap + 2.0 * (p['T_cap'] - self.T)))
        a = min(1.0, max(0.0, (X - p['X_lo']) / (p['X_hi'] - p['X_lo'])))
        L = min(p['L_lo'] + a * (p['L_hi'] - p['L_lo']), self.Lcap)

        # harvest request for the window ending at H
        H = 12.0 * (w + 1)
        if self.bound is None:
            f = 0.0
        elif H <= 84.0:
            f = 1.0 - p['X_hold'] / Xh
        elif H <= 96.0:
            f = 1.0 - p['X_96'] / Xh
        elif H <= 108.0:
            f = 0.5 if Xh >= p['X_108'] else 0.5 * (Xh - 50.0) / (p['X_108'] - 50.0)
        else:
            f = 0.5
        f = min(0.5, max(0.0, f))
        self.req_sum += f
        self.req_n += 1
        return (p['stir'], L, f)
