"""Supervisory controller for the Spirulina PBR (white-box design, run wb1). See controller.py."""
import math


class Controller:
    P = dict(rpm=70.0, imax=1800.0, ia=450.0, ib=1400.0, ramp=1000.0, odt=2.0, bleed=9,
             od_floor=0.15, mu_nom=0.031, turb_alpha=0.02, deadbeat=1)

    def __init__(self, params=None):
        self.p = dict(self.P)
        if params:
            self.p.update(params)
        self.turb = None
        self.c = 1.0
        self.od = None
        self.rpm_act = 50.0
        self.I = 300.0
        self.req_sum = 0.0
        self.n_req = 0
        self.mu = None

    def _invert(self, turb):
        y = turb / (250.0 * self.c ** (-1.0 / 3.0))
        y = min(y, 15.0)
        return y / max(0.25, 1.0 - 0.05 * y)

    def _mu_guess(self, od, I):
        """Light-limited growth rate from the design light model (path-integrated Haldane)."""
        X = od * 300.0
        rpm = self.rpm_act
        ks = rpm * 0.004
        cs = self.c ** (-1.0 / 3.0)
        L = 0.02 / 0.3
        fr = 0.0
        sg = sq = 0.0
        for i in range(8):
            zz = (i + 0.5) / 8.0 * L
            qr = I * 0.4 * math.exp(-(0.5 + 0.2 * X + ks) * zz)
            qt = qr + I * 0.4 * math.exp(-(0.2 + 0.25 * X + ks) * zz) + I * 0.2 * math.exp(-(0.05 + 0.06 * X + ks) * zz)
            g, q = cs * qr, cs * qt
            fr += g / (100.0 + g + q * q / 2500.0)
            sg += g
            sq += q
        fr /= 8.0
        rm, tm = sg / 8.0, sq / 8.0
        fint = rm / (100.0 + rm + tm * tm / 2500.0)
        w = 0.5 + 0.5 * min(max((rpm - 50.0) / 150.0, 0.0), 1.0)
        fi = min(1.0, (w * fint + (1.0 - w) * fr) / 0.5)
        rt = 1.0 - 0.35 / (1.0 + math.exp(-0.12 * (rpm - 100.0)))
        return self.p["mu_nom"] * fi * rt

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
        rpm = p["rpm"]
        self.rpm_act = 0.9 * self.rpm_act + 0.1 * rpm
        r = self.rpm_act
        if t % 10 == 0 or self.mu is None:
            self.mu = self._mu_guess(od, self.I)
        mu = self.mu
        stick = od * 0.05 * max(0.1, 1.0 - r / 250.0)
        shear = max(0.0, (r - 80.0) / 120.0) ** 2
        br = 0.5 * shear * math.sqrt(self.c) + 0.005 * math.sqrt(max(self.c - 1.0, 0.0))
        self.c = max(1.0, self.c + (stick - br - mu * (self.c - 1.0)) * 0.02)

        It = min(p["imax"], p["ia"] + p["ib"] * od)
        if It > self.I:
            self.I = min(It, self.I + p["ramp"] * 0.02)
        else:
            self.I = It

        # harvest: the pump applies the mean of requests over steps (600k-599 .. 600k]
        j = t % 600
        if j == 1 or t == 0:
            self.req_sum, self.n_req = 0.0, 0
        k = (t + 599) // 600          # index of the event this step belongs to (t=600 -> 1)
        rem = (600 - j) % 600          # steps still to come after this one in the interval
        if t == 0:
            k, rem = 1, 599
        od_pred = od * math.exp(mu * rem * 0.02)
        if k >= p["bleed"]:
            ftar = 0.5
        else:
            ftar = 1.0 - p["odt"] / od_pred if od_pred > p["odt"] else 0.0
        ftar = min(ftar, 1.0 - p["od_floor"] / max(od_pred, 1e-6))
        ftar = min(0.5, max(0.0, ftar))
        if p["deadbeat"]:
            n_total = 600 if k > 1 else 600
            f = (ftar * n_total - self.req_sum) / (rem + 1)
            f = min(0.5, max(0.0, f))
        else:
            f = ftar
        self.req_sum += f
        self.n_req += 1
        return rpm, self.I, f
