import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        # Use a more sophisticated light control based on lux and light umol
        light_control = np.clip(p["light_umol"] - 1000 * np.exp(-card["lux"] / 30), 0, p["light_umol"])
        light_control = np.clip(light_control, 0, p["light_umol"])

        # Use a more sophisticated stir control based on conductivity and stir rpm
        stir_control = np.clip(p["conductivity"] * p["stir_threshold"] / 50, 0, p["conductivity_threshold"])
        stir_control = stir_control / (p["conductivity"] / 50 + 1)
        stir_control = stir_control * p["stir"]

        # Use a more sophisticated fraction control based on growth rate and light umol
        frac = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0) * light_control / p["light_umol"], 0.0, p["cap"])
        frac = frac * stir_control

        return p["stir"], light_control, frac


DEFAULTS = {"stir": 65.0, "light_umol": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0, "conductivity_threshold": 150, "stir_threshold": 150,
            "light_umol_threshold": 1200, "light_control_coefficient": 0.7, "stir_control_coefficient": 0.8}
