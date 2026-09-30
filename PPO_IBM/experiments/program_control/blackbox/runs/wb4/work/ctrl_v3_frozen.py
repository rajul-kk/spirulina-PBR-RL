"""wb4 supervisory controller for the Spirulina photobioreactor.

Design (see LAB_NOTEBOOK.md):
- Density observer: smoothed turbidity, with the nephelometer's multiple-scattering
  saturation inverted and the filament-clumping under-read corrected by a clump-state observer.
  The observer integrates the plant's sticking/shear-breakup kinetics from the estimated density
  and the stirring actually delivered, and daughter cells reset clumps at the estimated growth rate.
- Stirring: 70 rpm while thin, 80 rpm in the working range (just below the shear-repair
  and fatigue penalties), and 140 rpm only in very dense broth, where clumps must be broken.
- Light: 500 + 5 umol per mg/L up to 1700 umol (the chiller still holds the broth near 35-36 C).
  Increases are ramped at 120 umol/h to limit photo-shock. Light is backed off if the broth
  reads hot.
- Harvest: per 12 h interval, chooses the interval-mean fraction that brings the density
  predicted at the event back to Xt = 800 mg/L. Earlier requests in the interval are corrected
  so the mean lands on target. Endgame: the targets drop to 0.85*Xt and 0.45*Xt at events 8-9,
  and the fraction is 0.5 at events 10-11, because biomass left at 144 h is never harvested.
  Endgame harvests are skipped for a culture thinner than 40 mg/L, so a tiny culture is not
  taken below the extinction floor.
"""
import math

DEFAULTS = dict(
    Xt=800.0,          # standing density to return to after each harvest (mg/L)
    n_end=2,           # the last n events harvest at the maximum fraction
    late=(0.85, 0.45), # target multipliers for the two events before the endgame
    min_X_end=40.0,    # no endgame harvest below this density (extinction guard)
    Imax=1700.0, I0=500.0, I_slope=5.0, ramp=120.0,
    rpm_lo=70.0, rpm_mid=80.0, rpm_hi=140.0, x_mid=200.0, x_hi=1100.0,
    T_hot=38.0,        # measured broth temperature above which light is backed off
    g0=0.025,          # prior specific growth rate (1/h)
)

N_EVENTS = 11          # harvest events at 12, 24, ... 132 h
INTERVAL = 600         # control steps per harvest interval
DT_H = 0.02


def _num(obs, key, default):
    try:
        v = float(obs.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.turb = None       # smoothed turbidity (NTU)
        self.c = 1.0           # clump-size observer
        self.rpm_act = 50.0    # delivered stirring (actuator lag model)
        self.I = None          # current light command
        self.S = 0.0           # sum of harvest requests in this interval
        self.m = 0             # number of requests in this interval
        self.last_frac = 0.0   # fraction applied at the previous event
        self.hist = []         # (t, log X) samples for the growth-rate estimate
        self.g = p["g0"]
        self.X = 0.0
        self.t_prev = -1

    def _density(self):
        r = min(self.turb / 250.0, 15.0)
        od = r / max(1.0 - 0.05 * r, 0.25)
        return 300.0 * od * self.c ** (1.0 / 3.0)

    def act(self, obs):
        p = self.p
        t = int(_num(obs, "t", self.t_prev + 1))
        self.t_prev = t
        turb = max(_num(obs, "turbidity_ntu", self.turb if self.turb is not None else 100.0), 0.0)

        # --- density observer ---
        if self.turb is None:
            self.turb = turb
        else:
            self.turb += 0.08 * (turb - self.turb)
        if t > 0 and t % INTERVAL == 1 and self.last_frac > 0.0:
            self.turb *= (1.0 - self.last_frac)   # the broth was just diluted
            self.hist = []
        X = self._density()
        if turb > 990.0:                           # nephelometer saturated
            X = max(X, 1500.0)
        self.X = X

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
                g = sxy / sxx / DT_H
                self.g = min(max(0.7 * self.g + 0.3 * g, -0.02), 0.06)

        # --- stirring ---
        if X < p["x_mid"]:
            rpm = p["rpm_lo"]
        elif X < p["x_hi"]:
            rpm = p["rpm_mid"]
        else:
            rpm = p["rpm_hi"]
        self.rpm_act = 0.9 * self.rpm_act + 0.1 * rpm
        ra = self.rpm_act
        stick = X / 300.0 * 0.05 * max(0.1, 1.0 - ra / 250.0)
        brk = (0.5 * max(0.0, (ra - 80.0) / 120.0) ** 2 * self.c ** 0.5
               + 0.005 * max(self.c - 1.0, 0.0) ** 0.5)
        self.c = max(1.0, self.c + (stick - brk - max(self.g, 0.0) * (self.c - 1.0)) * DT_H)

        # --- light ---
        target = min(p["Imax"], p["I0"] + p["I_slope"] * X)
        T = _num(obs, "temp_c", 35.0)
        if self.I is None:
            self.I = min(target, 400.0)
        if T > p["T_hot"]:
            target = min(target, self.I - 50.0 * (T - p["T_hot"]))
        if target > self.I:
            self.I = min(target, self.I + p["ramp"] * DT_H)
        else:
            self.I = max(target, self.I - 300.0 * DT_H)
        self.I = min(max(self.I, 0.0), 2000.0)

        # --- harvest ---
        nxt = INTERVAL if t == 0 else (t if t % INTERVAL == 0 else (t // INTERVAL + 1) * INTERVAL)
        ev = nxt // INTERVAL
        rem_steps = nxt - t + 1
        if ev > N_EVENTS:
            F = 0.0
        else:
            Xp = X * math.exp(max(self.g, 0.0) * rem_steps * DT_H)
            first_end = N_EVENTS + 1 - p["n_end"]
            if ev >= first_end:
                F = 0.5 if Xp > p["min_X_end"] else 0.0
            else:
                xt = p["Xt"]
                j = first_end - ev
                if j <= len(p["late"]):
                    xt *= p["late"][len(p["late"]) - j]
                F = min(0.5, max(0.0, 1.0 - xt / max(Xp, 1.0)))
            if turb > 950.0:
                F = 0.5
        need = F * (self.m + rem_steps) - self.S
        frac = min(0.5, max(0.0, need / rem_steps))
        self.S += frac
        self.m += 1
        if t > 0 and t % INTERVAL == 0:
            self.last_frac = self.S / self.m
            self.S, self.m = 0.0, 0
        return float(rpm), float(self.I), float(frac)
