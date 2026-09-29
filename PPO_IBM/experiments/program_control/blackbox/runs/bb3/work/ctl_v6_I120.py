"""Supervisory controller for the Spirulina photobioreactor (run bb3).

Design (from pilot batches, see LAB_NOTEBOOK.md):
  * stir at the minimum (50 rpm): least shear, ~25% faster growth than 120 rpm; dense-start batches
    (saturated turbidity at t=0) do slightly better at 200 rpm.
  * light follows estimated biomass: thin cultures are photo-inhibited, so light is set to keep the
    depth-averaged irradiance near the optimum (~250 umol) -> ~370 umol at DW 20, 1800 (cap) above DW ~200.
    Light is backed off if the broth runs hot.
  * harvest: grow with no harvest until DW reaches the hold level, then take just the growth above it
    each 12 h; from 108 h take the maximum 0.5 per harvest (biomass left at 144 h is worth nothing).
  * biomass is estimated from turbidity with a calibration fitted to lab assays at 50 rpm (turbidity
    reads progressively low as filaments clump); saturated turbidity (>950 NTU) means a very dense
    culture -> harvest the maximum.
"""
import math

STIR = 50.0
STIR_DENSE = 200.0      # batches that start with saturated turbidity (very dense inoculum)
DENSE_NTU = 700.0
L_MIN, L_MAX = 300.0, 1800.0
I_OPT = 120.0           # depth-averaged irradiance target, umol/m2/s
K_EXT = 0.0398          # light extinction per (mg/L) (fitted)
T_HOT, T_SLOPE = 38.5, 250.0   # above T_HOT, cut light by T_SLOPE umol per C
HOLD_EARLY = 550.0      # post-harvest DW target (mg/L) for harvests up to 84 h
HOLD_96 = 450.0         # post-harvest target at the 96 h harvest
ENDGAME_H = 108.0       # from this harvest on take the max fraction
SMALL_AT_108 = 150.0    # ...unless the culture is still this thin at 108 h
MU_PRED = 0.015         # growth assumed when projecting DW to the harvest instant (1/h)
MIN_DW_HARV = 60.0      # never harvest below this DW before 120 h
INTERVAL = 600          # control steps per 12 h harvest interval
DT_H = 0.02


def dw_from_ntu(ntu, hour):
    # fitted on 295 lab assays from 50 rpm batches (rms 14%): ln DW = a + b ln N + c ln^2 N + d h
    ln = math.log(max(ntu, 1.0))
    ln0 = 6.3969  # ln(600): few assays above this, continue linearly with the local slope
    if ln > ln0:
        base = 2.19386 - 0.018671 * ln0 + 0.126417 * ln0 * ln0 + 1.5987 * (ln - ln0)
    else:
        base = 2.19386 - 0.018671 * ln + 0.126417 * ln * ln
    return math.exp(base + 0.0029769 * hour)


class Controller:
    def __init__(self, params=None):
        self.ntu = None
        self.temp = None
        self.req_sum = 0.0
        self.k = -1
        self.t = -1
        self.dense = None

    def act(self, obs):
        t = obs.get('t', None)
        t = self.t + 1 if t is None else int(t)
        self.t = t
        hour = t * DT_H
        ntu = float(obs.get('turbidity_ntu', 0.0) or 0.0)
        temp = float(obs.get('temp_c', 35.0) or 35.0)
        if self.ntu is None:
            self.ntu, self.temp = ntu, temp
            self.dense = ntu > DENSE_NTU
        else:
            self.ntu += (ntu - self.ntu) / 50.0
            self.temp += (temp - self.temp) / 25.0
        sat = self.ntu > 950.0
        if self.dense:
            # 200 rpm keeps filaments dispersed: use the 120-200 rpm calibration (no clumping drift)
            X = 0.84 * max(self.ntu, 1.0) ** 1.106
        else:
            X = dw_from_ntu(self.ntu, hour)
        if sat:
            X = max(X, 1700.0)
        X = max(X, 5.0)

        kx = K_EXT * X
        light = I_OPT * kx / (1.0 - math.exp(-kx))
        if self.temp > T_HOT:
            light -= T_SLOPE * (self.temp - T_HOT)
        light = min(L_MAX, max(L_MIN, light))

        k = t // INTERVAL
        if k != self.k:
            self.k = k
            self.req_sum = 0.0
        pos = t - k * INTERVAL
        remaining = INTERVAL - pos
        h_harv = 12.0 * (k + 1)
        Xp = X * math.exp(MU_PRED * remaining * DT_H)
        if sat:
            target = 0.5
        elif h_harv >= ENDGAME_H - 1e-6:
            if h_harv < ENDGAME_H + 1.0 and Xp < SMALL_AT_108:
                target = 0.0
            else:
                target = 0.5
        else:
            hold = HOLD_96 if h_harv >= 95.0 else HOLD_EARLY
            target = 0.0 if Xp < MIN_DW_HARV else min(0.5, max(0.0, 1.0 - hold / Xp))
        need = (target * (pos + remaining) - self.req_sum) / remaining
        req = min(0.5, max(0.0, need))
        self.req_sum += req
        return (STIR_DENSE if self.dense else STIR, light, req)
