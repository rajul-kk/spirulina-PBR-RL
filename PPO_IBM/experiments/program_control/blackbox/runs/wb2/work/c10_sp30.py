"""Supervisory controller for the Spirulina photobioreactor (run wb2).

Design (derived from the plant model and pilot trials, see LAB_NOTEBOOK.md):
  * Biomass estimate: turbidity inverted for the nephelometer's multiple-scattering saturation and
    for filament clumping, with clumping predicted by an internal mean-field clump model driven by
    the estimated OD and the stirring history. Filtered; rescaled on each harvest by the fraction
    requested (the pump counter itself is too noisy to difference).
  * Light: the growth-optimal panel intensity for the estimated density (precomputed from the
    light-attenuation / Haldane model), corrected for clump self-shading, capped by a thermal
    limit, ramped gently so cells are not photo-shocked.
  * Stirring: fixed gentle rate (75 rpm); above ~90 rpm the shear/fatigue tax outweighs mixing
    benefits. Short 200 rpm bursts break filament clumps (which shade cells) whenever the
    predicted mean clump size passes a threshold.
  * Harvest: aim the post-harvest density at a setpoint using a growth forecast to the next
    harvest; re-planned during the interval so the interval MEAN hits the target; the last
    harvests take the maximum because biomass left at the end is worth nothing.
"""
import math
import numpy as np

DT = 0.02
D = 20e-3 / 0.3
ZQ = (np.arange(8) + 0.5) / 8.0 * D


def _f_light(X, L, rpm, Ks=100.0, Ki=2500.0):
    """Relative light response (0..1); X and L broadcastable arrays."""
    ks = 0.004 * rpm
    z = ZQ
    X3 = X[..., None]
    L3 = L[..., None]
    r = L3 * 0.4 * np.exp(-(0.5 + 0.2 * X3 + ks) * z)
    tot = r + L3 * 0.4 * np.exp(-(0.2 + 0.25 * X3 + ks) * z) + L3 * 0.2 * np.exp(-(0.05 + 0.06 * X3 + ks) * z)
    fr = (r / (Ks + r + tot ** 2 / Ki)).mean(axis=-1)
    rm, tm = r.mean(axis=-1), tot.mean(axis=-1)
    fi = rm / (Ks + rm + tm ** 2 / Ki)
    w = 0.5 + 0.5 * min(max((rpm - 50.0) / 150.0, 0.0), 1.0)
    Ip = math.sqrt(Ks * Ki)
    fmax = 0.4 * Ip / (2 * Ks + 0.4 * Ip)
    return np.clip((w * fi + (1 - w) * fr) / fmax, 0.0, 1.0)


class Controller:
    DEFAULTS = dict(
        rpm=75.0,          # stirring
        Lmax=1800.0,       # thermal cap on light
        L0=250.0,          # starting light (cells arrive acclimated to ~100-300)
        ramp_h=3.0,        # light ramp at start
        od_sp=3.0,         # post-harvest OD setpoint (true OD units)
        k_final=3,         # number of final harvests taken at the maximum
        mu_nom=0.034,      # nominal max growth rate used in forecasts (1/h)
        fmax=0.5,
        t_hot=41.0,        # measured temp above which the light cap is trimmed (safety only;
                           # the temp sensor can read ~2 C high from drift)
        dL_up=3.0,         # max light increase per step (umol/m2/s)
        burst_rpm=200.0,   # de-clumping burst: stir hard when the predicted mean clump size
        c_hi=2.0,          # exceeds c_hi, until it is back below c_lo
        c_lo=1.2,
    )

    def __init__(self, params=None):
        p = dict(self.DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        # light lookup: optimal effective intensity (before clump shading) vs biomass
        self.XG = np.exp(np.linspace(math.log(3.0), math.log(4000.0), 70))
        Lg = np.arange(50.0, 4001.0, 25.0)
        F = _f_light(self.XG[:, None], Lg[None, :], p["rpm"])
        self.LOPT = Lg[np.argmax(F, axis=1)]
        self.FOPT = F.max(axis=1)
        self.logXG = np.log(self.XG)
        self.od = None          # filtered true-OD estimate
        self.C = 1.0            # predicted mean clump size
        self.rpm_a = 50.0
        self.pump_prev = 0.0
        self.hsum = 0.0         # sum of harvest requests in the current interval
        self.hn = 0
        self.L = p["L0"]
        self.hist = []          # (hour, ln od) since the last harvest, every 50 steps
        self.Lmax_dyn = p["Lmax"]
        self.g_last = 0.02
        self.g_cache = 0.02
        self.last_frac = 0.0
        self.burst = False

    def _od_from_turb(self, turb):
        y = max(turb, 0.0) / 250.0
        k = self.C ** (-1.0 / 3.0)
        den = k - 0.05 * y
        if den <= 0.2 * k:
            return 4.0 * y / k
        return y / den

    def _growth_forecast(self):
        """Specific growth rate (1/h) for the next-harvest forecast."""
        X = self.od * 300.0
        s = self.C ** (-1.0 / 3.0)
        lx = math.log(min(max(X, 3.0), 3999.0))
        fI = float(np.interp(lx, self.logXG, self.FOPT))
        Lopt = float(np.interp(lx, self.logXG, self.LOPT))
        if self.L * s < Lopt * 0.98:
            fI = float(_f_light(np.array([X]), np.array([self.L * s]), self.p["rpm"])[0])
        g_model = self.p["mu_nom"] * 0.78 * fI
        if len(self.hist) >= 5:
            t0, l0 = self.hist[max(0, len(self.hist) - 8)]
            t1, l1 = self.hist[-1]
            if t1 - t0 >= 4.0:
                g_obs = min(max((l1 - l0) / (t1 - t0), 0.0), 0.05)
                return 0.5 * g_model + 0.5 * g_obs
        return g_model

    def act(self, obs):
        p = self.p
        t = int(obs["t"])
        turb = float(obs["turbidity_ntu"])
        temp = float(obs["temp_c"])

        # harvest: the event is applied during step t=600k, so the obs at t=600k+1 is post-harvest.
        # The pump_L sensor is noisy (+-2% jitter, +-5% drift), so use the fraction we requested
        # (the interval mean the pump applies) rather than differencing the counter.
        if t > 1 and t % 600 == 1 and self.od is not None:
            frac = min(max(self.last_frac, 0.0), 0.95)
            self.od *= (1.0 - frac)
            self.hist = []

        # mean-field clump model on the current estimate
        rpm = self.rpm_a
        od_c = self.od if self.od is not None else 0.2
        stick = od_c * 0.05 * DT * max(0.1, 1.0 - rpm / 250.0)
        shear = max(0.0, (rpm - 80.0) / 120.0) ** 2
        brk = (0.5 * shear * math.sqrt(self.C) + 0.005 * math.sqrt(max(self.C - 1.0, 0.0))) * DT
        self.C = max(1.0, self.C + stick - brk + self.g_last * DT * (1.0 - self.C))

        # OD estimate
        od_m = self._od_from_turb(turb)
        if self.od is None:
            self.od = od_m
        else:
            a = 0.04 if t > 25 else 0.3
            innov = max(min(od_m - self.od, 0.25 * self.od + 0.02), -0.25 * self.od - 0.02)
            self.od += a * innov
        X = max(self.od * 300.0, 1.0)
        if t % 50 == 0:
            self.hist.append((t * DT, math.log(max(self.od, 1e-4))))
            if len(self.hist) > 12:
                self.hist.pop(0)

        # light
        s = self.C ** (-1.0 / 3.0)
        lx = math.log(min(max(X, 3.0), 3999.0))
        Lopt = float(np.interp(lx, self.logXG, self.LOPT)) / s
        if temp > p["t_hot"]:
            self.Lmax_dyn = max(1000.0, self.Lmax_dyn - 2.0)
        elif temp < p["t_hot"] - 1.0:
            self.Lmax_dyn = min(p["Lmax"], self.Lmax_dyn + 1.0)
        L = min(Lopt, self.Lmax_dyn)
        th = t * DT
        if th < p["ramp_h"]:
            L = min(L, p["L0"] + (L - p["L0"]) * th / p["ramp_h"])
        if L > self.L:
            L = min(L, self.L + p["dL_up"])
        self.L = L

        # harvest planning (forecast refreshed every 10 steps)
        if t % 10 == 0:
            self.g_cache = self._growth_forecast()
        g = self.g_cache
        self.g_last = g
        j = t % 600
        ev = (t - 1) // 600 + 1 if t > 0 else 1        # upcoming event index, 1..11
        if j == 1 and t > 1:
            self.hsum, self.hn = 0.0, 0
        steps_left = (600 - j) if j != 0 else 0
        if ev > 11 - p["k_final"]:
            target = p["fmax"]
        else:
            Xp = X * math.exp(g * steps_left * DT)
            target = min(max(1.0 - p["od_sp"] * 300.0 / Xp, 0.0), p["fmax"])
        n_total = 601 if ev == 1 else 600
        remaining = max(n_total - self.hn, 1)
        req = min(max((target * n_total - self.hsum) / remaining, 0.0), p["fmax"])
        self.hsum += req
        self.hn += 1
        if j == 0 and t > 0:
            self.last_frac = self.hsum / self.hn
        rpm_cmd = p["rpm"]
        if self.C > p["c_hi"]:
            self.burst = True
        elif self.C < p["c_lo"]:
            self.burst = False
        if self.burst:
            rpm_cmd = p["burst_rpm"]
        self.rpm_a = 0.9 * self.rpm_a + 0.1 * rpm_cmd
        return float(rpm_cmd), float(L), float(req)
