"""cand_003 (Claude): cand_002 + per-batch extremum seeking on light.

Strains differ per batch (light half-saturation and inhibition constants), so no fixed light
schedule suits all of them. Each 12 h interval is split into six 2 h blocks with light at
centre +d, -d, +d, -d, ... ; the growth rate (log-turbidity slope) of + blocks minus - blocks
estimates dGrowth/dLight, and the centre moves along it. The density schedule from cand_002 is
the starting centre and the ES offset is added on top. Skipped in the end-of-batch window.
"""
import math

import numpy as np

from process_state import ProcessState

DEFAULTS = {"stir": 55.0, "od_post": 0.55, "cap": 0.5, "min_od_to_harvest": 0.6,
            "terminal_events": 2, "light_thin": 900.0, "light_mid": 1300.0, "light_dense": 1600.0,
            "dither": 150.0, "es_gain": 3.0e7, "offset_lim": 500.0}
EVENT_STEPS = 600
EPISODE_STEPS = 7200
BLOCK = 100          # 2 h
SETTLE = 15          # skip the first 18 min of each block (sensor EMA lag)


def ntu_per_od(hours, turb):
    return math.exp(6.787 - 0.0018 * hours - 0.2943 * math.log(max(turb, 1.0)))


class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState()
        self.offset = 0.0
        self.block_logs = []           # log turbidity samples in the current block
        self.slopes = {+1: [], -1: []}

    def _es_update(self, t):
        """Called at block boundaries: record the finished block's slope; at the end of an
        interval, step the offset along the estimated gradient."""
        sign = +1 if ((t - 1) // BLOCK) % 2 == 0 else -1
        ys = self.block_logs[SETTLE:]
        if len(ys) > 20:
            x = np.arange(len(ys), dtype=float)
            slope = float(np.polyfit(x, ys, 1)[0]) / 0.02          # 1/h
            self.slopes[sign].append(slope)
        self.block_logs = []
        if t % EVENT_STEPS == 0:
            if self.slopes[+1] and self.slopes[-1]:
                grad = (np.mean(self.slopes[+1]) - np.mean(self.slopes[-1])) / (2 * self.p["dither"])
                step = float(np.clip(self.p["es_gain"] * grad, -100.0, 100.0))   # umol per interval
                self.offset = float(np.clip(self.offset + step, -self.p["offset_lim"], self.p["offset_lim"]))
            self.slopes = {+1: [], -1: []}

    def act(self, obs):
        p = self.p
        c = self.ps.update(obs)
        turb = self.ps.turb_fast
        od = turb / ntu_per_od(c["hours"], turb)
        t = int(obs["t"])

        base = p["light_thin"] if od < 0.3 else p["light_mid"] if od < 0.8 else p["light_dense"]

        last_event = (EPISODE_STEPS - 1) // EVENT_STEPS * EVENT_STEPS
        first_terminal = last_event - (p["terminal_events"] - 1) * EVENT_STEPS
        if t >= first_terminal - EVENT_STEPS:
            return p["stir"], float(np.clip(base + self.offset, 300.0, 1800.0)), p["cap"]

        if t > 0 and t % BLOCK == 0:
            self._es_update(t)
        # harvest events dilute turbidity: skip the block containing one
        if t % EVENT_STEPS >= 2:
            self.block_logs.append(math.log(max(float(obs["turbidity_ntu"]), 1e-3)))
        sign = +1 if (t // BLOCK) % 2 == 0 else -1
        light = float(np.clip(base + self.offset + sign * p["dither"], 300.0, 1800.0))

        g = float(np.clip(c["growth_per_h"], -0.05, 0.08))
        od_at_event = od * math.exp(g * c["hours_to_harvest"])
        if od_at_event < p["min_od_to_harvest"]:
            frac = 0.0
        else:
            frac = float(np.clip(1.0 - p["od_post"] / od_at_event, 0.0, p["cap"]))
        return p["stir"], light, frac
