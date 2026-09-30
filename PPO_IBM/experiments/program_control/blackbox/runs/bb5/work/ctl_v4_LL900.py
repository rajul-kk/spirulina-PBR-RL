import math

# Productivity curve P(X) (mg/L/h) fitted from pilot lab assays at light >= 1100
_PX = [0.0, 50.0, 100.0, 150.0, 250.0, 400.0, 600.0, 1000.0, 1300.0, 1800.0, 2300.0, 2900.0, 4000.0]
_PY = [0.0, 1.1, 2.3, 3.5, 5.5, 6.3, 7.0, 7.2, 7.0, 5.0, 2.5, 0.0, -1.0]
# Nephelometer response (NTU at ~48 h) vs dry weight, fitted to lab assays; falls ~0.23%/h with clumping
_NX = [0.0, 50.0, 150.0, 300.0, 550.0, 850.0, 1200.0, 1700.0, 2300.0, 3000.0, 5000.0]
_NY = [0.0, 37.7, 111.9, 195.3, 310.1, 400.0, 445.0, 508.4, 603.2, 674.6, 800.0]


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            a = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + a * (ys[i] - ys[i - 1])
    return ys[-1]


class Controller:
    STIR = 50.0
    L_LOW = 900.0      # light at very low density (photoinhibition below ~50 mg/L)
    X_LOW = 40.0
    X_FULL = 200.0
    L_MAX = 1300.0     # more light only heats the tank; growth saturates ~1100-1400
    X_HOLD = 1050.0    # pre-harvest density to hold if the culture is that dense
    TH96 = 700.0       # end-game: take 0.5 at 96 h harvest if predicted X above this
    TH108 = 350.0
    VOL = 20.0         # L, from pump volumes
    X_TRUST = 400.0    # turbidity trusted below this estimated density
    SAT_NTU = 900.0
    KN = 0.005
    SMIN = 0.25

    def __init__(self, params=None):
        if params:
            for k, v in params.items():
                setattr(self, k, v)
        self.xm = None       # model estimate of dry weight, mg/L
        self.tf = None       # filtered turbidity
        self.pump_last = 0.0
        self.saturated0 = False
        self.req = 0.0       # fraction requested for the current interval
        self.req_k = -1

    def ratio(self, h):
        return max(0.58, 0.77 - 0.0017 * h)

    def P(self, x):
        return _interp(x, _PX, _PY)

    def predict(self, x, hours):
        n = int(hours / 0.5) + 1
        dt = hours / n
        for _ in range(n):
            x = x + dt * self.P(x)
        return x

    def act(self, obs):
        t = obs['t']
        h = t * 0.02
        turb = obs['turbidity_ntu']
        pump = obs['pump_L']
        if self.xm is None:
            self.tf = turb
            if turb >= self.SAT_NTU:
                self.saturated0 = True
                self.xm = 2500.0
            else:
                self.xm = _interp(turb / 1.11, _NY, _NX)
            self.pump_last = pump
        # model propagation
        self.xm += 0.02 * self.P(self.xm)
        # harvest detection from the pump totaliser
        dp = pump - self.pump_last
        if dp > 0.3:
            frac = min(0.5, dp / self.VOL)
            self.xm *= (1.0 - frac)
            self.tf = turb
        self.pump_last = pump
        self.tf += 0.1 * (turb - self.tf)
        if not (self.saturated0 and h < 12.5) and self.tf < self.SAT_NTU:
            fac = max(0.6, 1.0 - 0.0023 * (h - 48.0))
            npred = _interp(self.xm, _NX, _NY) * fac
            slope = (_interp(self.xm * 1.05 + 1.0, _NX, _NY) - _interp(self.xm * 0.95 - 1.0, _NX, _NY)) * fac / (self.xm * 0.1 + 2.0)
            self.xm += self.KN * (self.tf - npred) / max(slope, self.SMIN)
            if self.xm < 1.0:
                self.xm = 1.0
        X = self.xm

        # light schedule by density
        if X <= self.X_LOW:
            light = self.L_LOW
        elif X >= self.X_FULL:
            light = self.L_MAX
        else:
            a = (X - self.X_LOW) / (self.X_FULL - self.X_LOW)
            light = self.L_LOW + a * (self.L_MAX - self.L_LOW)
        light = min(light, self.L_MAX)

        # harvest request, decided once per 12 h interval (pump uses the mean)
        # decide a few steps into each interval, once the previous harvest has registered
        k = int((t - 5) // 600) if t >= 5 else -1   # harvest at end = 12(k+1) h
        if k >= 0 and k != self.req_k:
            self.req_k = k
            hn = 12 * (k + 1)
            left = hn - h
            xp = self.predict(X, left)
            if hn >= 120:
                f = 0.5
            elif hn == 108:
                f = 0.5 if xp >= self.TH108 else 0.0
            elif hn == 96:
                f = 0.5 if xp >= self.TH96 else 0.0
            else:
                f = 0.0
            if f < 0.5 and xp > self.X_HOLD:
                f = min(0.5, 1.0 - self.X_HOLD / xp)
            if self.saturated0 and hn == 12:
                f = 0.5
            self.req = f
        return (self.STIR, light, self.req)
