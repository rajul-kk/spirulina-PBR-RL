"""Supervisory controller for the Spirulina PBR (white-box design, run wb1).

Structure
- Observer: an internal mean-field model tracks the mean filament clump size (which makes the
  nephelometer read low by clump^-1/3) and the photo-acclimation state. OD is estimated from a
  filtered turbidity reading corrected for modelled clumping and multiple-scattering saturation.
- Stir: fixed low-shear speed (clumps vs the shear repair tax), per the design study.
- Light: surface intensity scheduled on estimated OD (optimum of the Haldane light curve over
  the light path), increases rate-limited so cells stay acclimated (no photo-shock).
- Harvest: hold the culture at a productive target OD by harvesting the growth; never harvest
  while the culture is small; bleed at the maximum rate over the final events because standing
  biomass at the end is worth nothing.
Only numpy/math/collections; no I/O.
"""
import math


class Controller:
    P = dict(rpm_lo=70.0, rpm_hi=70.0, od_rpm=99.0, imax=1800.0, ia=450.0, ib=1400.0,
             ramp=200.0, odt=2.0, bleed=9, od_min_harvest=0.6, turb_alpha=0.02)

    def __init__(self, params=None):
        self.p = dict(self.P)
        if params:
            self.p.update(params)
        self.turb = None
        self.c = 1.0          # modelled mean clump mass
        self.od = None
        self.rpm_act = 50.0   # modelled delivered stir (actuator lag 0.9/0.1 per step)
        self.I = 300.0
        self.req_sum = 0.0    # sum of harvest fractions requested this interval

    def _invert(self, turb):
        y = turb / (250.0 * self.c ** (-1.0 / 3.0))
        y = min(y, 15.0)
        return y / max(0.25, 1.0 - 0.05 * y)

    def act(self, obs):
        p = self.p
        t = int(obs["t"])
        z = float(obs["turbidity_ntu"])
        if self.turb is None:
            self.turb = z
        else:
            self.turb += p["turb_alpha"] * (z - self.turb)
        od = self._invert(self.turb)
        self.od = od

        # stir
        rpm = p["rpm_lo"] if od < p["od_rpm"] else p["rpm_hi"]
        self.rpm_act = 0.9 * self.rpm_act + 0.1 * rpm
        r = self.rpm_act
        # clump observer (per-step, dt = 0.02 h)
        stick = od * 0.05 * max(0.1, 1.0 - r / 250.0)
        shear = max(0.0, (r - 80.0) / 120.0) ** 2
        br = 0.5 * shear * math.sqrt(self.c) + 0.005 * math.sqrt(max(self.c - 1.0, 0.0))
        mu_guess = 0.02
        self.c = max(1.0, self.c + (stick - br - mu_guess * (self.c - 1.0)) * 0.02)

        # light
        It = min(p["imax"], p["ia"] + p["ib"] * od)
        if It > self.I:
            self.I = min(It, self.I + p["ramp"] * 0.02)
        else:
            self.I = It

        # harvest
        j = t % 600
        if j == 1 or t == 0:
            self.req_sum = 0.0
        k = t // 600 + 1 if t % 600 else t // 600
        if k >= p["bleed"]:
            f = 0.5
        elif od > max(p["odt"], p["od_min_harvest"]):
            f = min(0.5, max(0.0, 1.0 - p["odt"] / od))
        else:
            f = 0.0
        self.req_sum += f
        return rpm, self.I, f
