"""Supervisory controller for the Spirulina photobioreactor (run bb4).

Design (from pilot batches, see LAB_NOTEBOOK.md):
  * stir held at the 50 rpm minimum: shear at 120-200 rpm cut growth by 30-45 %.
  * biomass X (mg/L) is inferred from turbidity with an empirical nephelometer
    model fitted to the lab dry-weight assays:
        NTU = S * (1 - exp(-g * c(t) * X / S)),  S=535, g=0.85,
        c(t) = 1 - 0.002 * max(t_h - 10, 0)   (filament clumping)
    and cross-checked against an optimistic growth-model prediction so a
    faulty/high turbidity reading can never drive heavy over-harvest.
  * light follows density (thin cultures are photo-inhibited, dense ones are
    light-limited): L = min(2000, 800 + 5 X), backed off if broth > 40.5 C.
  * harvest: hold the culture near 550 mg/L (productivity plateau ~7-8 mg/L/h
    from ~400 mg/L up), predicting X at the next pump event with the fitted
    growth model; the pump takes the mean request over the interval, so the
    request is metered to hit the planned fraction.  The last three harvests
    (108, 120, 132 h) take the maximum 0.5 to recover standing crop, since
    biomass left at 144 h is worth nothing.
"""
import math


class Controller:
    P = dict(stir=50.0,
             # nephelometer model
             S=535.0, g=0.85, a=0.002, t_clump=10.0,
             # growth model (depth-averaged Haldane light response)
             mumax=0.0842, Ks=698.3, Ki=721.9, kx=0.00906, m=0.0008,
             opt=1.3,           # optimism factor for the plausibility bound
             target=550.0,
             L0=800.0, Lk=5.0, Lmax=2000.0, Tmax=40.5,
             end_int=(9, 10, 11), end_frac=0.5, end_min=150.0,
             steps=600, alpha=0.01, clip=0.25, s2=120.0)

    def __init__(self, params=None):
        self.p = dict(self.P)
        if params:
            self.p.update(params)
        self.k = 0
        self.ntu = None
        self.cur_int = 0
        self.req_sum = 0.0
        self.light = self.p['L0']
        self.F = 0.0
        self.X = 0.0
        self.Xhi = None      # optimistic open-loop bound on biomass
        self.T = None

    # ---- models -------------------------------------------------------
    def est(self, ntu, h):
        p = self.p
        c = max(0.5, 1.0 - p['a'] * max(0.0, h - p['t_clump']))
        S = p['S']
        y = min(max(ntu, 0.0), 0.93 * S) / S
        return -S / (p['g'] * c) * math.log(1.0 - y)

    def mu(self, X, L):
        p = self.p
        kx = max(p['kx'] * X, 1e-9)
        f = 0.0
        for i in range(6):
            I = L * math.exp(-kx * (i + 0.5) / 6.0)
            f += I / (p['Ks'] + I + I * I / p['Ki'])
        return p['mumax'] * f / 6.0 - p['m']

    def Lfun(self, X):
        p = self.p
        return min(p['Lmax'], p['L0'] + p['Lk'] * X)

    def predict(self, X, hours):
        n = max(1, int(hours / 0.5))
        dt = hours / n
        for _ in range(n):
            X *= math.exp(self.mu(X, self.Lfun(X)) * dt)
        return X

    # ---- control step -------------------------------------------------
    def act(self, obs):
        p = self.p
        k = self.k
        self.k += 1
        h = k * 0.02
        N = p['steps']
        idx = k // N
        if idx != self.cur_int:
            # the pump has just removed the mean requested fraction
            Fm = self.req_sum / N
            if self.ntu is not None:
                self.ntu *= (1.0 - Fm)
            if self.Xhi is not None:
                self.Xhi *= (1.0 - Fm)
            self.cur_int = idx
            self.req_sum = 0.0

        try:
            n = float(obs.get('turbidity_ntu', float('nan')))
        except (TypeError, ValueError):
            n = float('nan')
        if n == n and n >= 0.0:
            if self.ntu is None:
                self.ntu = n
            else:
                lo = self.ntu * (1.0 - p['clip'])
                hi = self.ntu * (1.0 + p['clip']) + 2.0
                self.ntu += p['alpha'] * (min(max(n, lo), hi) - self.ntu)
        ntu = self.ntu if self.ntu is not None else 0.0
        Xe = self.est(ntu, h)

        if self.Xhi is None:
            if k >= 25:              # anchor after ~0.5 h of filtering
                self.Xhi = max(Xe, 5.0) * 1.3
            X = Xe
        else:
            self.Xhi *= math.exp(max(0.0, self.mu(self.Xhi, self.light)) * p['opt'] * 0.02)
            self.Xhi = max(self.Xhi, 0.0)
            X = min(Xe, self.Xhi)
        self.X = X

        # light
        L = self.Lfun(X)
        try:
            T = float(obs.get('temp_c', 35.0))
        except (TypeError, ValueError):
            T = 35.0
        if T != T:
            T = 35.0
        self.T = T if self.T is None else self.T + 0.05 * (T - self.T)
        if self.T > p['Tmax']:
            L = min(L, max(p['L0'], self.light - 2.0))
        elif L > self.light + 20.0:
            L = self.light + 20.0
        L = min(p['Lmax'], max(0.0, L))
        self.light = L

        # harvest
        pos = k - idx * N
        left = N - pos
        harvest_no = idx + 1
        if harvest_no > 11:
            F = 0.0
        elif harvest_no in p['end_int'] and X > p['end_min']:
            F = p['end_frac']
        else:
            if pos % 10 == 0 or self.F == 0.0:
                Xp = self.predict(X, left * 0.02)
                self.F = min(0.5, max(0.0, 1.0 - p['target'] / Xp)) if Xp > 0 else 0.0
            F = self.F
        r = (F * N - self.req_sum) / left
        r = min(0.5, max(0.0, r))
        self.req_sum += r
        s = p['stir'] if h < 12.0 else p['s2']
        return (s, L, r)
