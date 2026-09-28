import math

class Controller:
    """v1: NTU-based density estimate with clumping correction; grow -> hold -> dump; temp-capped light."""
    DEF = dict(stir=80.0, Lmax=1800.0, Lmin=600.0, Tcap=38.5, Xpost=400.0, ndump=3,
               Xfloor=60.0, K=1230.0, kc=8e-6, cmin=0.5, gproj=4.0, Lgain=40.0)

    def __init__(self, params=None):
        self.p = dict(self.DEF); self.p.update({'Lmax': 1400.0, 'stir': 60.0, 'Xpost': 250.0, 'ndump': 4.0, 'Tcap': 38.5}); self.p.update(params or {})
        self.ntu = None; self.T = None; self.c = 1.0
        self.L = 1200.0; self.k = -1; self.S = 0.0; self.X = 0.0

    def act(self, obs):
        p = self.p; dt = 0.02
        t = int(obs['t'])
        n = float(obs['turbidity_ntu']); T = float(obs['temp_c'])
        a = dt / 1.0  # ~1 h EMA
        self.ntu = n if self.ntu is None else self.ntu + a * (n - self.ntu)
        self.T = T if self.T is None else self.T + (dt / 0.5) * (T - self.T)
        sat = min(self.ntu, 950.0) / 1000.0
        X = -p['K'] * math.log(1.0 - sat) / self.c
        self.X = X
        # clumping model: factor decays with integrated biomass at low stir
        if p['stir'] < 110:
            kc = p['kc'] * (110.0 - p['stir']) / 30.0 if p['stir'] > 80 else p['kc']
            self.c = max(p['cmin'], self.c - kc * X * dt)
        # light: integral action on temperature cap
        self.L += p['Lgain'] * dt * (p['Tcap'] - self.T) * 10.0
        self.L = min(p['Lmax'], max(p['Lmin'], self.L))
        # harvest: interval k covers steps [600k, 600k+600); harvest k+1 at its end (hour 12(k+1))
        k = t // 600
        if k != self.k:
            self.k = k; self.S = 0.0
        hk = k + 1                      # index of the harvest this interval feeds (1..11 valid)
        rem_h = (600 - (t - 600 * k)) * dt
        if hk > 11:
            fdes = 0.0
        elif hk > 11 - p['ndump']:
            fdes = 0.5
        else:
            Xproj = X + p['gproj'] * rem_h
            fdes = 0.0 if Xproj <= p['Xpost'] else 1.0 - p['Xpost'] / Xproj
        # safety floor: never take the tank below Xfloor
        Xproj = X + p['gproj'] * rem_h
        if Xproj > 0:
            fdes = min(fdes, max(0.0, 1.0 - p['Xfloor'] / Xproj))
        fdes = min(0.5, max(0.0, fdes))
        steps_left = 600 - (t - 600 * k)
        r = (fdes * 600.0 - self.S) / steps_left
        r = min(0.5, max(0.0, r))
        self.S += r
        return (p['stir'], self.L, r)
