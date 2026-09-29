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
        light_control = np.clip(p["light_umol"] - card["lux"] / 30, 0, p["light_umol"])
        light_control = np.clip(np.clip(light_control, 0, p["light_umol"]), 0, 1000)

        # Use a more sophisticated stir control based on conductivity and stir rpm
        stir_control = np.clip(p["conductivity_threshold"] - p["conductivity"] * 50, 0, p["conductivity_threshold"])
        stir_control = stir_control / (p["conductivity"] / 50 + 1)
        stir_control = stir_control * p["stir_threshold"]

        frac = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"])
        frac = frac * stir_control

        return p["stir"], light_control, frac

