import math

# ---- tunables ----
STIR = 50.0
L_MIN, L_SLOPE, L_MAX = 500.0, 6.0, 1800.0
L_OFF = 450.0
HOLD_EARLY = 500.0      # post-harvest target DW (mg/L) for harvests at <= 84 h
HOLD_96 = 450.0         # post-harvest target at 96 h
ENDGAME_H = 108.0       # from this harvest on, take the max fraction
MU_PRED = 0.015         # growth assumed when projecting DW to the harvest instant
MIN_DW_HARV = 60.0      # never harvest below this before the endgame
INTERVAL = 600          # steps per 12 h harvest interval


def dw_from_ntu(ntu):
    ntu = max(ntu, 1.0)
    return 0.84 * ntu ** 1.106


class Controller:
    def __init__(self, params=None):
        self.ntu = None
        self.req_sum = 0.0
        self.req_n = 0
        self.k = -1

    def act(self, obs):
        t = int(obs.get('t', 0))
        ntu = float(obs.get('turbidity_ntu', 0.0))
        if self.ntu is None:
            self.ntu = ntu
        else:
            self.ntu += (ntu - self.ntu) / 50.0
        sat = self.ntu > 950.0
        X = dw_from_ntu(self.ntu)
        if sat:
            X = max(X, 1600.0)

        light = min(L_MAX, max(L_MIN, L_OFF + L_SLOPE * X))

        # harvest interval bookkeeping
        k = t // INTERVAL
        if k != self.k:
            self.k = k
            self.req_sum = 0.0
            self.req_n = 0
        pos = t - k * INTERVAL
        remaining = INTERVAL - pos
        h_harv = 12.0 * (k + 1)
        Xp = X * math.exp(MU_PRED * remaining * 0.02)
        if h_harv >= ENDGAME_H - 1e-6:
            target = 0.5
        else:
            hold = HOLD_96 if h_harv >= 95.0 else HOLD_EARLY
            if Xp < MIN_DW_HARV:
                target = 0.0
            else:
                target = min(0.5, max(0.0, 1.0 - hold / Xp))
            if sat:
                target = 0.5
        need = (target * (self.req_n + remaining) - self.req_sum) / remaining
        req = min(0.5, max(0.0, need))
        self.req_sum += req
        self.req_n += 1
        return (STIR, light, req)
