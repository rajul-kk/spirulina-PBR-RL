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
    L_LOW = 800.0      # light at very low density (photoinhibition below ~50 mg/L)
    X_LOW = 40.0
    X_FULL = 200.0
    L_MAX = 1500.0     # more light only heats the tank; growth saturates ~1100-1400
    X_HOLD = 1050.0    # pre-harvest density to hold if the culture is that dense
    VOL = 20.0         # L, from pump volumes
    X_TRUST = 400.0    # turbidity trusted below this estimated density
    SAT_NTU = 900.0
    KN = 0.005
    T_GUARD = 38.5
    CAL_EXP = 0.3
    X_HI = 600.0
    L_HI = 1900.0
    SMIN = 0.25

    def __init__(self, params=None):
        if params:
            for k, v in params.items():
                setattr(self, k, v)
        self.xm = None       # model estimate of dry weight, mg/L
        self.tf = None       # filtered turbidity
        self.saturated0 = False
        self.req = 0.0       # fraction requested for the current interval
        self.req_k = -1
        self.temp_f = None
        self.cal = 1.0
        self.req_sum = 0.0
        self.req_n = 0
        self.pending = None
        self.light = self.L_LOW

    @staticmethod
    def _num(v, default):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        if v != v or v in (float('inf'), float('-inf')):
            return default
        return v

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
        t = int(self._num(obs.get('t'), 0))
        h = t * 0.02
        turb = self._num(obs.get('turbidity_ntu'), self.tf if self.tf is not None else 100.0)
        temp = self._num(obs.get('temp_c'), self.temp_f if self.temp_f is not None else 35.0)
        if turb < 0.0:
            turb = 0.0
        if self.temp_f is None:
            self.temp_f = temp
        self.temp_f += 0.02 * (temp - self.temp_f)
        if self.xm is None:
            self.tf = turb
            if turb >= self.SAT_NTU:
                self.saturated0 = True
                self.xm = 2500.0
            else:
                self.xm = _interp(turb / 1.11, _NY, _NX)
        # model propagation
        self.xm += 0.02 * self.P(self.xm)
        # harvest accounting: at each 12 h boundary the pump removes the mean of the fractions
        # we requested over the interval (pump_L is too noisy to detect it step by step)
        if t > 0 and t % 600 == 0 and self.req_n > 0:
            frac = min(0.5, max(0.0, self.req_sum / self.req_n))
            self.req_sum = 0.0
            self.req_n = 0
            if frac > 0.005:
                n_before = _interp(self.xm, _NX, _NY)
                self.xm *= (1.0 - frac)
                n_after = _interp(self.xm, _NX, _NY)
                self.pending = (self.tf, n_before, n_after, t + 10)
        if self.pending is not None:
            tf_before, n_before, n_after, t_done = self.pending
            if t < t_done:
                return self._out(light_hold=True)
            # dilution does not undo clumping: turbidity drops ~in proportion, unlike the
            # concave NTU(X) table; partially rescale the table (exponent from replay fits)
            if tf_before > 5.0 and n_before > 1.0 and n_after > 1.0:
                meas = min(1.0, max(0.2, turb / tf_before))
                expd = n_after / n_before
                self.cal *= min(1.6, max(0.6, meas / expd)) ** self.CAL_EXP
                self.cal = min(2.0, max(0.5, self.cal))
            self.tf = turb
            self.pending = None
        self.tf += 0.1 * (turb - self.tf)
        if not (self.saturated0 and h < 12.5) and self.tf < self.SAT_NTU:
            fac = max(0.6, 1.0 - 0.0023 * (h - 48.0))
            npred = _interp(self.xm, _NX, _NY) * fac * self.cal
            slope = (_interp(self.xm * 1.05 + 1.0, _NX, _NY) - _interp(self.xm * 0.95 - 1.0, _NX, _NY)) * fac * self.cal / (self.xm * 0.1 + 2.0)
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
        if X > self.X_HI:
            light = self.L_HI
        # overheating guard: the LEDs are the main heat load beyond the thermostat
        if self.temp_f > self.T_GUARD:
            light *= max(0.4, 1.0 - (self.temp_f - self.T_GUARD) / 2.5)

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
                f = (xp - 200.0) / 320.0
            elif hn == 96:
                f = (xp - 400.0) / 640.0
            elif hn == 84:
                f = (xp - 750.0) / 1060.0
            else:
                f = 0.0
            if xp > self.X_HOLD:
                f = max(f, 1.0 - self.X_HOLD / xp)
            f = min(0.5, max(0.0, f))
            if self.saturated0 and hn == 12:
                f = 0.5
            self.req = f
        self.light = light
        return self._out()

    def _out(self, light_hold=False):
        self.req_sum += self.req
        self.req_n += 1
        return (self.STIR, self.light, self.req)
