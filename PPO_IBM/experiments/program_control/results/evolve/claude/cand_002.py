"""cand_002 (Claude): cand_001 + end-of-batch harvest, gentler stirring, density-scheduled light.

- Biomass left in the tank at 144 h is never counted. With ~1.3x regrowth per 12 h interval,
  a unit of biomass is worth more harvested now than carried into either of the last two
  events, so both take the 0.5 cap (the env averages frac over each interval, so the whole
  interval must ask for it).
- Probes: growth falls ~35% from 65 to 130 rpm and is flat 50-65, so stir 55.
- Light optimum rises with density (~900 umol thin, ~1300 mid, ~1600+ dense); capped at 1600
  because the tank passes 36 C above that.
"""
import math

import numpy as np

from process_state import ProcessState

DEFAULTS = {"stir": 55.0, "od_post": 0.55, "cap": 0.5, "min_od_to_harvest": 0.6,
            "terminal_events": 2, "light_thin": 900.0, "light_mid": 1300.0, "light_dense": 1600.0}
EVENT_STEPS = 600
EPISODE_STEPS = 7200


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

        light = p["light_thin"] if od < 0.3 else p["light_mid"] if od < 0.8 else p["light_dense"]

        t = int(obs["t"])
        last_event = (EPISODE_STEPS - 1) // EVENT_STEPS * EVENT_STEPS           # step 6600
        first_terminal = last_event - (p["terminal_events"] - 1) * EVENT_STEPS
        if t >= first_terminal - EVENT_STEPS:     # inside an interval that ends at a terminal event
            return p["stir"], light, p["cap"]

        g = float(np.clip(c["growth_per_h"], -0.05, 0.08))
        od_at_event = od * math.exp(g * c["hours_to_harvest"])
        if od_at_event < p["min_od_to_harvest"]:
            frac = 0.0
        else:
            frac = float(np.clip(1.0 - p["od_post"] / od_at_event, 0.0, p["cap"]))
        return p["stir"], light, frac
