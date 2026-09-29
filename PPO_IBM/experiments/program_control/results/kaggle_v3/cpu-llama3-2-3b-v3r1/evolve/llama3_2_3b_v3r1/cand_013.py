import numpy as np
from process_state import ProcessState


class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        if "lux" in obs and "light_umol" in obs:
            light_control = np.clip(1200 - (card["lux"] / 30) * (obs["light_umol"] / 2000), 0, 1200)
        else:
            light_control = np.clip(obs["light_umol"], 0, p["light_umol"])

        if "conductivity" in obs and "stir" in obs:
            stir_control = np.clip(p["conductivity_threshold"] * p["stir_threshold"] * (obs["conductivity"] / 50), 0, p["conductivity_threshold"])
        else:
            stir_control = 0

        frac = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"])
        frac = frac * stir_control

        return p["stir"], light_control, frac

DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0, "conductivity_threshold": 150, "light_umol_threshold": 1200,
            "stir_threshold": 150, "stir_control_coefficient": 0.8, "light_control_coefficient": 0.7}
