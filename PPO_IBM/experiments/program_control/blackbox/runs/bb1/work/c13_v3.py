import math
# turbidity -> dry-weight calibration (stir 50), fitted on bb1 assays:
# ln NTU = c0 + c1*l + c2*t + c3*l^2 + c4*t*l,  l = ln DW (mg/L), t = hour
C = (-2.22876449, 1.86843690, 1.61463661e-03, -9.31409675e-02, -7.22231493e-04)
P = dict(stir=50.0, L_lo=900.0, L_hi=1500.0, X_lo=150.0, X_hi=400.0, T_cap=37.5, L_min=800.0,
         X_hold=750.0, X_96=420.0, X_108=150.0)

def ntu_to_dw(n, t):
    c0, c1, c2, c3, c4 = C
    n = max(n, 1.0)
    a = c3; b = c1 + c4 * t; c = c0 + c2 * t - math.log(n)
    lv = -b / (2 * a)                      # vertex (turbidity saturates beyond it)
    disc = b * b - 4 * a * c
    if disc <= 0: return math.exp(lv)
    l = (-b + math.sqrt(disc)) / (2 * a)   # rising branch root
    return math.exp(min(l, lv))

class Controller:
    """v3: stir 50; density-scheduled light with temperature servo ceiling; harvest policy from a
    dynamic programme on the fitted growth model (hold <= X_hold early, drain from 96-108 h)."""
    def __init__(self, params=None):
        self.p = dict(P)
        if params: self.p.update(params)
        self.ntu = None; self.T = None; self.Lcap = 2000.0
    def act(self, obs):
        p = self.p
        hr = obs['t'] * 0.02
        n = obs['turbidity_ntu']; T = obs['temp_c']
        if self.ntu is None or abs(n - self.ntu) > 0.3 * self.ntu and hr % 12 < 0.1:
            self.ntu = n                      # (re)initialise, e.g. right after a harvest
        else:
            self.ntu += 0.03 * (n - self.ntu)  # ~0.7 h filter
        self.T = T if self.T is None else self.T + 0.02 * (T - self.T)
        X = ntu_to_dw(self.ntu, hr)
        # light
        self.Lcap = min(2000.0, max(p['L_min'], self.Lcap + 2.0 * (p['T_cap'] - self.T)))
        a = min(1.0, max(0.0, (X - p['X_lo']) / (p['X_hi'] - p['X_lo'])))
        L = min(p['L_lo'] + a * (p['L_hi'] - p['L_lo']), self.Lcap)
        # harvest: the pump applies the mean request over the window ending at the next 12 h mark
        H = 12.0 * (math.floor(hr / 12.0 + 1e-9) + 1)
        if H <= 84.0:
            f = 1.0 - p['X_hold'] / X
        elif H <= 96.0:
            f = 1.0 - p['X_96'] / X
        elif H <= 108.0:
            f = 0.5 if X >= p['X_108'] else 0.5 * (X - 50.0) / (p['X_108'] - 50.0)
        else:
            f = 0.5
        f = min(0.5, max(0.0, f))
        return (p['stir'], L, f)
