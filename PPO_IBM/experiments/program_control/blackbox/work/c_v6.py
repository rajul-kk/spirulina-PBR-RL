import math


class Controller:
    """Supervisory controller for the Spirulina PBR (v5, final).

    - Stirring: fixed low (shear hurts growth; >=110 rpm avoids clumping but grows ~40% slower).
    - Light: fixed Lset; a slow guard lowers light only if the temperature reading rises Tguard_rise
      above the batch's own early baseline (thermostat saturation), never below Lfloor.
    - Density estimate: saturating nephelometer inverse X = -K ln(1 - NTU/1000), divided by a
      clumping factor c that decays with integrated biomass at low stir (dc/dt = -kc X).
    - Harvest: grow unharvested until X exceeds Xpost, then take the 12 h growth (hold X_post after
      each harvest); dump 0.5 on the last ndump harvests; never take the tank below Xfloor.
      Requests are made so that the 12 h MEAN equals the desired fraction.
    """
    DEF = dict(stir=50.0, Lset=1600.0, Lfloor=1000.0, Tguard_rise=2.5, Tabs=38.0,
               Xpost=350.0, ndump=4, Xfloor=60.0, Xend=(300.0, 120.0, 60.0, 35.0), K=1230.0, kc=8e-6, cmin=0.5, gproj=4.0)

    def __init__(self, params=None):
        self.p = dict(self.DEF); self.p.update({'Lset': 1900.0, 'Lfloor': 1400.0, 'Tabs': 37.0, 'Tguard_rise': 0.0}); self.p.update(params or {})
        self.ntu = None; self.T = None; self.c = 1.0; self.X = 0.0
        self.L = self.p['Lset']; self.k = -1; self.S = 0.0
        self.Tb_sum = 0.0; self.Tb_n = 0; self.Tb = None; self.tn = 0

    def act(self, obs):
        p = self.p; dt = 0.02
        t = int(obs.get('t', self.tn)); self.tn = t + 1
        try:
            n = float(obs.get('turbidity_ntu', float('nan')))
        except (TypeError, ValueError):
            n = float('nan')
        try:
            T = float(obs.get('temp_c', float('nan')))
        except (TypeError, ValueError):
            T = float('nan')
        if not (n == n): n = self.ntu if self.ntu is not None else 0.0
        if not (T == T): T = self.T if self.T is not None else 35.0
        self.ntu = n if self.ntu is None else self.ntu + dt * (n - self.ntu)          # ~1 h EMA
        self.T = T if self.T is None else self.T + (dt / 1.0) * (T - self.T)
        sat = min(max(self.ntu, 0.0), 950.0) / 1000.0
        X = -p['K'] * math.log(1.0 - sat) / self.c
        self.X = X
        if p['stir'] < 110:
            kc = p['kc'] if p['stir'] <= 80 else p['kc'] * (110.0 - p['stir']) / 30.0
            self.c = max(p['cmin'], self.c - kc * X * dt)
        # temperature baseline: mean reading over hours 3-9
        hr = t * dt
        if 3.0 <= hr < 9.0:
            self.Tb_sum += T; self.Tb_n += 1
        elif hr >= 9.0 and self.Tb is None and self.Tb_n > 0:
            self.Tb = self.Tb_sum / self.Tb_n
        Tlim = p['Tabs']
        if self.T > Tlim:
            self.L -= 20.0 * (self.T - Tlim) * dt * 10.0
        else:
            self.L += 100.0 * dt
        self.L = min(p['Lset'], max(p['Lfloor'], self.L))
        # harvest
        k = t // 600
        if k != self.k:
            self.k = k; self.S = 0.0
        hk = k + 1
        steps_left = 600 - (t - 600 * k)
        rem_h = steps_left * dt
        Xproj = X + p['gproj'] * rem_h
        if hk > 11:
            fdes = 0.0
        elif hk > 11 - p['ndump']:
            # endgame draw-down: keep at least Xend[j] after harvest (DP on my grey-box model: small,
            # still-exponential cultures are worth more left in the tank until the last harvests)
            j = max(0, len(p['Xend']) - (12 - hk))
            fdes = 0.0 if Xproj <= p['Xend'][j] else 1.0 - p['Xend'][j] / Xproj
        else:
            fdes = 0.0 if Xproj <= p['Xpost'] else 1.0 - p['Xpost'] / Xproj
        floor = p['Xfloor'] if hk <= 11 - p['ndump'] else min(p['Xfloor'], p['Xend'][-1])
        if Xproj > 0:
            fdes = min(fdes, max(0.0, 1.0 - floor / Xproj))
        fdes = min(0.5, max(0.0, fdes))
        r = (fdes * 600.0 - self.S) / steps_left
        r = min(0.5, max(0.0, r))
        self.S += r
        return (p['stir'], self.L, r)
