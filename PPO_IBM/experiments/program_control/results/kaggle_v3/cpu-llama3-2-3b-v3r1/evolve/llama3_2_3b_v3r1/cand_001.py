import numpy as np
from process_state import ProcessState


class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        # Use more sophisticated light control based on lux and light umol
        light_control = min(p["light_umol"], 1000 - card["lux"] / 30)
        light_control = np.clip(light_control, 0, p["light_umol"])

        # Use a more sophisticated stir control based on conductivity and stir rpm
        stir_control = min(p["conductivity_threshold"], p["stir_threshold"] * p["conductivity"] / 50)
        stir_control = np.clip(stir_control, 0, p["conductivity_threshold"])
        stir_control = stir_control / (p["conductivity"] / 50 + 1)
        stir_control = stir_control * p["stir"]

        frac = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"])
        frac = frac * stir_control

        return p["stir"], light_control, frac


DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0, "conductivity_threshold": 150, "light_umol_threshold": 1200,
            "stir_threshold": 150, "stir_control_coefficient": 0.8, "light_control_coefficient": 0.7}
