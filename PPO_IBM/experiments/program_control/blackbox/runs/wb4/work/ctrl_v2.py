"""wb4 supervisory controller, v2: DP-shaped harvest schedule."""
import math

DEFAULTS = dict(
    Xt=560.0,        # standing density to keep after each harvest (mg/L)
    n_end=2,         # last n events harvest at the maximum
    late=(0.85, 0.45), # post-harvest target multipliers for the two events before the endgame
    min_X_end=40.0,
    Imax=1700.0,     # light ceiling (umol)
    I0=500.0,        # light at zero density
    I_slope=5.0,     # extra umol per mg/L
    ramp=120.0,      # max light increase, umol per hour
    rpm_lo=70.0, rpm_mid=80.0, rpm_hi=140.0,
    x_mid=200.0, x_hi=800.0,
    T_hot=38.0,      # back light off above this broth temperature
    g0=0.025,        # prior growth rate (1/h) for harvest prediction
    min_end_X=150.0, # no endgame harvest if culture thinner than this (mg/L)
)


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.turb = None
        self.c = 1.0          # clump observer
        self.rpm_act = 50.0   # actuator model (0.9/0.1 EMA)
        self.I = None
        self.S = 0.0          # harvest command sum this interval
        self.m = 0
        self.X = None
        self.hist = []        # (t, logX) samples for growth estimate
        self.g = p["g0"]
        self.last_event_t = 0

    # density estimate (mg/L) from smoothed turbidity, inverting saturation and clumping
    def _X(self):
        r = self.turb / 250.0
        r = min(r, 15.0)
        od = r / max(1.0 - 0.05 * r, 0.25)
        return 300.0 * od * self.c ** (1.0 / 3.0)

    def act(self, obs):
        p = self.p
        t = int(obs.get("t", 0))
        turb = float(obs["turbidity_ntu"])
        if self.turb is None:
            self.turb = turb
        else:
            self.turb += 0.08 * (turb - self.turb)
        # a harvest just happened: the broth was diluted, snap the filter down
        if t > 0 and t % 600 == 1 and self.last_frac > 0.0:
            self.turb *= (1.0 - self.last_frac)
            self.hist = []
        X = self._X()
        if turb > 990.0:
            X = max(X, 1500.0)
        self.X = X

        # growth-rate estimate from log-density slope over the last ~6 h
        if t % 25 == 0:
            self.hist.append((t, math.log(max(X, 1.0))))
            if len(self.hist) > 13:
                self.hist.pop(0)
            if len(self.hist) >= 8:
                n = len(self.hist)
                mt = sum(h[0] for h in self.hist) / n
                my = sum(h[1] for h in self.hist) / n
                sxx = sum((h[0] - mt) ** 2 for h in self.hist)
                sxy = sum((h[0] - mt) * (h[1] - my) for h in self.hist)
                g = sxy / sxx / 0.02
                self.g = min(max(0.7 * self.g + 0.3 * g, -0.02), 0.06)

        # stirring schedule
        if X < p["x_mid"]:
            rpm = p["rpm_lo"]
        elif X < p["x_hi"]:
            rpm = p["rpm_mid"]
        else:
            rpm = p["rpm_hi"]
        self.rpm_act = 0.9 * self.rpm_act + 0.1 * rpm
        ra = self.rpm_act
        od = X / 300.0
        stick = od * 0.05 * max(0.1, 1.0 - ra / 250.0)
        brk = 0.5 * max(0.0, (ra - 80.0) / 120.0) ** 2 * self.c ** 0.5 + 0.005 * max(self.c - 1.0, 0.0) ** 0.5
        self.c = max(1.0, self.c + (stick - brk - max(self.g, 0.0) * (self.c - 1.0)) * 0.02)

        # light: target rises with density, ramped up slowly, cut if the broth overheats
        target = min(p["Imax"], p["I0"] + p["I_slope"] * X)
        T = float(obs.get("temp_c", 35.0))
        if T > p["T_hot"]:
            target = min(target, (self.I or target) - 50.0 * (T - p["T_hot"]))
        if self.I is None:
            self.I = min(target, 400.0)
        if target > self.I:
            self.I = min(target, self.I + p["ramp"] * 0.02)
        else:
            self.I = max(target, self.I - 300.0 * 0.02)
        self.I = min(max(self.I, 0.0), 2000.0)

        # harvest: choose the interval-mean fraction F, correct for what was already requested
        nxt = (t // 600 + 1) * 600 if t % 600 else t
        if t == 0:
            nxt = 600
        ev = nxt // 600
        rem_steps = nxt - t + 1
        if ev > 11:
            F = 0.0
        else:
            rem_h = rem_steps * 0.02
            Xp = X * math.exp(max(self.g, 0.0) * rem_h)
            first_end = 12 - p["n_end"]
            if ev >= first_end:
                F = 0.5 if Xp > p["min_X_end"] else 0.0
            else:
                xt = p["Xt"]
                j = first_end - ev          # 1 = the event just before the endgame
                if j <= len(p["late"]):
                    xt *= p["late"][len(p["late"]) - j]
                F = min(0.5, max(0.0, 1.0 - xt / max(Xp, 1.0)))
            if turb > 950.0:
                F = 0.5
        need = F * (self.m + rem_steps) - self.S
        frac = min(0.5, max(0.0, need / rem_steps))
        self.S += frac
        self.m += 1
        if t > 0 and t % 600 == 0:
            self.last_frac = self.S / self.m
            self.S, self.m = 0.0, 0
        elif t % 600 != 1:
            pass
        return float(rpm), float(self.I), float(frac)

    last_frac = 0.0
