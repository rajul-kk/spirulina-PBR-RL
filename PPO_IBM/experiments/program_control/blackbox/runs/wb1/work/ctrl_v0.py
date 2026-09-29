"""v0 probe controller: fixed stir, light scheduled on estimated OD with a ramp limit,
harvest toward a target OD, bleed-down at the end."""
import math

class Controller:
    def __init__(self, params=None):
        p = dict(rpm=70.0, odt=2.0, imax=1800.0, bleed=9, ramp=200.0, i0=300.0)
        if params: p.update(params)
        self.p = p
        self.turb = None
        self.I = p["i0"]
        self.hsum = 0.0

    def od_est(self):
        y = self.turb / 250.0
        # invert y = od/(1+0.05 od)
        return y / max(1e-6, 1.0 - 0.05 * y)

    def act(self, obs):
        p = self.p
        t = int(obs["t"])
        z = float(obs["turbidity_ntu"])
        self.turb = z if self.turb is None else self.turb + 0.05 * (z - self.turb)
        od = self.od_est()
        It = min(p["imax"], 450.0 + 1400.0 * od)
        self.I = min(It, self.I + p["ramp"] * 0.02) if It > self.I else It
        k = t // 600 + 1
        j = t % 600
        if j == 0 and t > 0:
            self.hsum = 0.0
        if k >= p["bleed"]:
            f = 0.5
        else:
            f = min(0.5, max(0.0, 1.0 - p["odt"] / od)) if od > p["odt"] else 0.0
        return p["rpm"], self.I, f
