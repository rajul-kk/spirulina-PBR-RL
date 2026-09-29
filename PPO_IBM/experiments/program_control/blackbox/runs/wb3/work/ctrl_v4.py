"""Supervisory controller v4 (wb3): grow-then-drain with an internal observer.

Observer: EMA-filtered turbidity -> OD, corrected for filament clumping and pigment
bleaching, both integrated internally from the plant equations (clump from OD and rpm,
pigment from path-mean light). Light holds the path-mean PAR near the Haldane optimum
(sqrt(Ks*Ki) ~ 500 umol) capped by I_MAX (thermal), with a slow ramp-up to avoid
photo-shock. Harvest: none while small, cap standing OD at OD_CAP, drain in the last
N_DRAIN intervals.
"""
import math

DEFAULTS = dict(
    STIR=75.0, STIR_HI=75.0, STIR_HI_OD=2.0,
    I_MIN=300.0, I_MAX=1700.0, TM_TARGET=500.0, RAMP_UP=4.0, RAMP_DN=10.0,
    OD_CAP=2.0, N_DRAIN=3, F_DRAIN=0.5, OD_MIN_HARVEST=0.25, MU_PRED=0.02, F_PRE=0.0, OD_MIN_DRAIN=0.08,
    TURB_ALPHA=0.04,
)

_Z = [(i + 0.5) / 8.0 * (20e-3 / 0.30) for i in range(8)]


def path_mean(I, X, rpm, clump):
    ks = rpm * 0.004
    kr, kb, kg = 0.5 + 0.2 * X + ks, 0.2 + 0.25 * X + ks, 0.05 + 0.06 * X + ks
    s = 0.0
    for z in _Z:
        s += 0.4 * math.exp(-kr * z) + 0.4 * math.exp(-kb * z) + 0.2 * math.exp(-kg * z)
    return I * s / 8.0 * clump ** (-1.0 / 3.0)


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.turb = None
        self.od = None
        self.clump = 1.0
        self.pig = 1.0
        self.rpm = 50.0
        self.light = None
        self.hsum = 0.0
        self.hn = 0

    def _od_from_turb(self, turb):
        od = self.od if self.od else turb / 250.0
        for _ in range(3):
            gain = 250.0 * (0.7 + 0.3 * self.pig) * self.clump ** (-1.0 / 3.0) / (1.0 + 0.05 * od)
            od = turb / gain
        return od

    def act(self, obs):
        p = self.p
        t = int(obs.get("t", 0))
        turb = float(obs.get("turbidity_ntu", 0.0))
        if self.turb is None:
            self.turb = turb
        else:
            self.turb += p["TURB_ALPHA"] * (turb - self.turb)
        od_meas = self._od_from_turb(self.turb)
        if self.od is not None and self.turb > 900.0:
            od_meas = max(od_meas, self.od)   # nephelometer near saturation: do not trust drops
        self.od = max(od_meas, 1e-3)
        od = self.od
        X = od * 300.0

        # stirring
        stir = p["STIR_HI"] if od > p["STIR_HI_OD"] else p["STIR"]
        self.rpm = 0.9 * self.rpm + 0.1 * stir
        rpm = self.rpm

        # light: path-mean target, thermal cap, slow ramp up
        lo, hi = 50.0, p["I_MAX"]
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if path_mean(mid, X, rpm, self.clump) < p["TM_TARGET"]:
                lo = mid
            else:
                hi = mid
        target = max(p["I_MIN"], min(p["I_MAX"], 0.5 * (lo + hi)))
        if self.light is None:
            self.light = min(target, p["I_MIN"])
        if target > self.light:
            self.light = min(target, self.light + p["RAMP_UP"])
        else:
            self.light = max(target, self.light - p["RAMP_DN"])
        tm = path_mean(self.light, X, rpm, self.clump)

        # internal clump / pigment observers (plant equations, per 0.02 h step)
        stick = od * 0.05 * max(0.1, 1.0 - rpm / 250.0)
        shear = max(0.0, (rpm - 80.0) / 120.0) ** 2
        brk = 0.5 * shear * math.sqrt(self.clump) + 0.005 * math.sqrt(max(self.clump - 1.0, 0.0))
        self.clump = max(1.0, self.clump + (stick - brk - (self.clump - 1.0) * p["MU_PRED"]) * 0.02)
        self.pig = min(1.0, max(0.2, self.pig + (-0.01 if tm > 1000.0 else 0.01) * 0.02))

        # harvest
        k = t // 600 + 1            # the event this step's request counts toward (at step 600k)
        left = 600 - (t % 600)      # steps remaining in this interval including this one
        hours_to_event = left * 0.02
        if k >= 12 - p["N_DRAIN"]:
            f_des = p["F_DRAIN"]
        else:
            od_ev = od * math.exp(p["MU_PRED"] * hours_to_event)
            f_des = max(0.0, 1.0 - p["OD_CAP"] / od_ev) if od_ev > p["OD_CAP"] else 0.0
            if k == 11 - p["N_DRAIN"]:
                f_des = max(f_des, p["F_PRE"])
        if k >= 12 - p["N_DRAIN"]:
            if od < p["OD_MIN_DRAIN"]:     # drain phase: only a near-empty tank is spared
                f_des = 0.0
        elif od < p["OD_MIN_HARVEST"]:
            f_des = 0.0
        if t % 600 == 0:
            self.hsum, self.hn = 0.0, 0
        need = f_des * 600.0 - self.hsum
        frac = min(0.5, max(0.0, need / left))
        self.hsum += frac
        self.hn += 1
        return stir, self.light, frac
