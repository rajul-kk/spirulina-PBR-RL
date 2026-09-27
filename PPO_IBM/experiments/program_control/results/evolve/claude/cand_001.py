"""cand_001 (Claude): calibrated OD estimate + predictive harvest to a post-harvest target.

- Turbidity-per-OD drifts from ~250 to ~130 NTU as cells clump and the sensor saturates, so
  the fixed 250 under-reads OD and the culture runs 1.3-1.9x target. K(h, turb) below was fitted
  offline on calibration batches (the simulator stand-in for dry-weight samples).
- Harvest removes the mean frac over the interval, so aim each step's frac at the OD predicted
  for the next event, sized to land at OD_POST after the harvest.
- Dense starts can use the full 0.5 fraction instead of the old 0.3 cap.
"""
import math

import numpy as np

from process_state import ProcessState

DEFAULTS = {"stir": 65.0, "light": 1400.0, "od_post": 0.55, "cap": 0.5, "min_od_to_harvest": 0.6}


def ntu_per_od(hours, turb):
    return math.exp(6.787 - 0.0018 * hours - 0.2943 * math.log(max(turb, 1.0)))


class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState()

    def act(self, obs):
        p = self.p
        c = self.ps.update(obs)
        turb = self.ps.turb_fast
        od = turb / ntu_per_od(c["hours"], turb)
        g = float(np.clip(c["growth_per_h"], -0.05, 0.08))
        od_at_event = od * math.exp(g * c["hours_to_harvest"])
        if od_at_event < p["min_od_to_harvest"]:
            frac = 0.0
        else:
            frac = float(np.clip(1.0 - p["od_post"] / od_at_event, 0.0, p["cap"]))
        return p["stir"], p["light"], frac
