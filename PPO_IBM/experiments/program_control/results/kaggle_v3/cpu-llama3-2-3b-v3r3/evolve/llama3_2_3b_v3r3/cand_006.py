import numpy as np
from process_state import ProcessState

class Controller:
    DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.3,
                "turb_per_od": 250.0}

    def __init__(self, params=None):
        self.p = dict(self.DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.t = 0
        self.harvest_frac = np.zeros((7200))

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip((card["od_est"] - p["setpoint"]) / (card["od_est"] + 1), 0.0, p["cap"]))
        self.harvest_frac[self.t] = frac
        if self.t % 600 == 0:
            self.harvest_frac[-600:] = 0.0  # reset the last 600 harvest fractions
            self.t = 0
        stir = np.clip(p["stir"] + self.t * 1, 50, 200)
        light = np.clip(p["light"] + self.t * 10, 0, 2000)
        return stir, light, self.harvest_frac[-600:].mean()

