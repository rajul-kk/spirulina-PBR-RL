"""Supervisory controller v3 for the Spirulina PBR (run bb2).

Plant facts used (from pilot batches, see LAB_NOTEBOOK.md):
- stir does not change growth measurably but shifts the turbidity reading -> stir held fixed at 100.
- turbidity is strongly sub-linear in biomass and reads low as the culture ages/densifies:
  ln NTU = ln(996*(1-exp(-X/1230))) - 0.056*(t/100) - 0.182*(t/100)*(X/1000)   (t in h, X in mg/L)
- growth is light-limited at high density (2000 umol/m2/s best above ~500 mg/L even though the
  tank heats to 38-40 C); at low density 1300 is as good as 2000 and cooler, and very dilute
  cultures (<60 mg/L) did best at ~800-1000.
- culture is lost if density falls to ~5 mg/L -> never harvest below 40 mg/L (estimated).
- no harvest at 144 h, so the end game dumps 50% at 108/120/132 h; before that the culture is
  held near the productivity optimum (~750 mg/L).
"""
import math

P = dict(
    STIR=100.0,
    A=996.0, D=1230.0, C1=0.056, C2=0.182,
    # light vs estimated density: piecewise linear through these (X mg/L, umol/m2/s) points
    L_PTS=((60.0, 1000.0), (150.0, 1300.0), (500.0, 2000.0)),
    T_GOV=40.0, T_GAIN=400.0, L_MIN=800.0,   # temperature governor (light cut above T_GOV)
    X_HOLD=750.0,
    END_K=8, X_POST_END=500.0, F_END=0.5,
    X_MIN_POST=40.0,
    X_CAP=3000.0,
    MU_GUESS=0.01,
)


class Controller:
    SPI = 600  # steps per 12 h harvest interval

    def __init__(self, params=None):
        self.p = dict(P)
        if params:
            self.p.update(params)
        self.k = 0
        self.ntu = None
        self.temp = None
        self.X = 0.0
        self.req_sum = 0.0

    def _ntu_model(self, X, hour):
        p = self.p
        th = hour / 100.0
        return p['A'] * (1.0 - math.exp(-X / p['D'])) * math.exp(-p['C1'] * th - p['C2'] * th * X / 1000.0)

    def _xest(self, ntu, hour):
        # model is increasing in X over the range of interest; bisection on [0, X_CAP]
        p = self.p
        lo, hi = 0.0, p['X_CAP']
        # the model peaks where exp(-X/D)/(D*(1-exp(-X/D))) = C2*th/1000; beyond it the reading
        # carries no information, so search only up to the peak
        g = p['C2'] * (hour / 100.0) / 1000.0 * p['D']
        if g > 0:
            hi = min(hi, p['D'] * math.log(1.0 + 1.0 / g))
        if ntu >= self._ntu_model(hi, hour):
            return hi
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self._ntu_model(mid, hour) < ntu:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def act(self, obs):
        p = self.p
        t = self.k
        self.k += 1
        hour = t * 0.02
        z = obs.get('turbidity_ntu', None)
        z = float(z) if z is not None and z == z else (self.ntu if self.ntu is not None else 0.0)
        tc = obs.get('temp_c', None)
        tc = float(tc) if tc is not None and tc == tc else 35.0
        if self.ntu is None:
            self.ntu, self.temp = z, tc
        else:
            self.ntu += 0.02 * (z - self.ntu)     # ~1 h filter
            self.temp += 0.02 * (tc - self.temp)
        if t % 5 == 0:
            self.X = self._xest(max(self.ntu, 0.0), hour)
        X = self.X

        # light: density schedule + temperature governor
        pts = p['L_PTS']
        if X <= pts[0][0]:
            L = pts[0][1]
        elif X >= pts[-1][0]:
            L = pts[-1][1]
        else:
            L = pts[-1][1]
            for (x0, l0), (x1, l1) in zip(pts[:-1], pts[1:]):
                if X <= x1:
                    L = l0 + (l1 - l0) * (X - x0) / (x1 - x0)
                    break
        if self.temp > p['T_GOV']:
            L = max(min(L, p['L_MIN']), L - p['T_GAIN'] * (self.temp - p['T_GOV']))

        # harvest request so that the 12 h interval mean equals the target fraction
        pos = t % self.SPI
        if pos == 0:
            self.req_sum = 0.0
        k_next = t // self.SPI + 1
        rem_h = (self.SPI - pos) * 0.02
        Xp = X * math.exp(p['MU_GUESS'] * rem_h)
        if k_next > p['END_K']:
            f = p['F_END']
        elif k_next == p['END_K']:
            f = 1.0 - p['X_POST_END'] / max(Xp, 1e-6)
        else:
            f = 1.0 - p['X_HOLD'] / max(Xp, 1e-6)
        f = max(0.0, min(0.5, f))
        if Xp * (1.0 - f) < p['X_MIN_POST']:
            f = max(0.0, 1.0 - p['X_MIN_POST'] / max(Xp, 1e-6))
        n_rem = self.SPI - pos
        r = (f * self.SPI - self.req_sum) / n_rem
        r = max(0.0, min(0.5, r))
        self.req_sum += r
        return (p['STIR'], L, r)
