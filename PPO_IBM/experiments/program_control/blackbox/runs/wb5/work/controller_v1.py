"""Supervisory controller for the Spirulina photobioreactor (wb5).

Design (from the design model + pilot trials):
- Biomass estimate: turbidity EMA, inverted for multiple-scattering saturation, corrected for
  filament clumping by an internal clump model (sticking ~ OD, breakup above 80 rpm, dilution
  by new cells) that mirrors the plant physics. Turbidity near the 1000 NTU ceiling = very dense.
- Light: follows biomass (thin cultures ~600 umol, dense >= ~1800), rate-limited upward to
  avoid photo-shock; capped so the thermostat keeps the broth near 37 C.
- Stir: moderate constant (below the ~100 rpm shear-repair penalty).
- Harvest: per-event biomass caps (grow to the productive density, hold it, then draw down
  hard over the final events); the requested fraction is steered within each 12 h interval so
  the interval MEAN equals the fraction wanted at the event. Never harvest a thin culture.
"""
import math

P = dict(
    stir=80.0,
    light_pts=((0.06, 600.0), (0.2, 750.0), (0.4, 1050.0), (0.6, 1400.0), (0.8, 1700.0), (1.2, 1800.0)),
    light_up_per_h=100.0,
    light_start=400.0,
    caps=(2.4, 2.4, 2.4, 2.4, 2.4, 2.7, 2.8, 1.7, 0.65, 0.0, 0.0),
    min_od_harvest=0.25,
    mu_nom=0.031,
    burst=None,          # (period_h, length_h, rpm) optional clump-breaking stir bursts
)

_FTAB = ((0.1, 0.98), (0.4, 0.93), (1.0, 0.80), (2.0, 0.59), (3.0, 0.46), (4.0, 0.37), (6.0, 0.24), (9.0, 0.13))


def _interp(x, pts):
    if x <= pts[0][0]:
        return pts[0][1]
    for (a, u), (b, v) in zip(pts, pts[1:]):
        if x <= b:
            return u + (v - u) * (x - a) / (b - a)
    return pts[-1][1]


class Controller:
    def __init__(self, params=None):
        self.p = dict(P)
        if params:
            self.p.update(params)
        self.turb = None
        self.clump = 1.0
        self.rpm_s = 50.0
        self.light = self.p['light_start']
        self.od = None
        self.hsum = 0.0
        self.last_event = 0

    def _od_estimate(self):
        r = self.turb / 250.0
        r = min(r, 15.0)
        od_app = r / max(0.25, 1.0 - 0.05 * r)
        od = od_app * self.clump ** (1.0 / 3.0)
        if self.turb > 950.0:           # sensor ceiling: culture is at least this dense
            od = max(od, 6.0)
        return od

    def act(self, obs):
        p = self.p
        t = int(obs.get('t', 0))
        dt = 0.02
        turb = float(obs.get('turbidity_ntu', 0.0))
        if not math.isfinite(turb):
            turb = self.turb if self.turb is not None else 0.0
        a = 0.05
        self.turb = turb if self.turb is None else self.turb + a * (turb - self.turb)
        od = self._od_estimate()
        self.od = od

        # stir (with optional bursts)
        stir = p['stir']
        th = t * dt
        b = p['burst']
        if b and th > 24.0 and (th % b[0]) < b[1] and th < 132.0:
            stir = b[2]
        self.rpm_s += 0.1 * (stir - self.rpm_s)

        # internal clump model (mirrors plant flocculation physics)
        mu = p['mu_nom'] * _interp(od, _FTAB)
        c = self.clump
        stick = od * 0.05 * max(0.1, 1.0 - self.rpm_s / 250.0)
        br = 0.5 * max(0.0, (self.rpm_s - 80.0) / 120.0) ** 2 * math.sqrt(c) + 0.005 * math.sqrt(max(c - 1.0, 0.0))
        self.clump = max(1.0, c + (stick - br - mu * (c - 1.0)) * dt)

        # light: follow biomass, rate-limited upward
        tgt = _interp(od, p['light_pts'])
        temp = float(obs.get('temp_c', 35.0))
        if math.isfinite(temp) and temp > 41.0:
            tgt = min(tgt, 1500.0)
        if tgt > self.light:
            self.light = min(tgt, self.light + p['light_up_per_h'] * dt)
        else:
            self.light = tgt

        # harvest: steer the interval mean to the fraction wanted at the coming event
        e = max(1, (t + 599) // 600)          # next event index (events at t = 600 e)
        if e > 11:
            return stir, self.light, 0.0
        if e != self.last_event:
            self.last_event = e
            self.hsum = 0.0
        start = 0 if e == 1 else 600 * (e - 1) + 1
        n_int = 600 * e - start + 1
        done = t - start
        remaining = n_int - done
        hours_left = (600 * e - t) * dt
        od_ev = od * math.exp(mu * hours_left)
        cap = p['caps'][e - 1]
        if od_ev < p['min_od_harvest']:
            want = 0.0
        else:
            cap = max(cap, 0.5 * p['min_od_harvest'])
            want = min(0.5, max(0.0, 1.0 - cap / od_ev))
        req = (want * n_int - self.hsum) / remaining
        req = min(0.5, max(0.0, req))
        self.hsum += req
        return stir, self.light, req
