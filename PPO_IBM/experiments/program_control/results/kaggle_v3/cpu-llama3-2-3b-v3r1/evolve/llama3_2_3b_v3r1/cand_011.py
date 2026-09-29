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
        if 'lux' in obs and 'light_umol' in obs:
            light_control = np.clip(p["light_umol"] - obs['lux'] / 30, 0, p["light_umol"])
            light_control = np.clip(light_control, 0, p["light_umol"])
            light_control = np.clip(light_control, 0, 1000)
        else:
            light_control = p["light_umol"]

        # Use a more sophisticated stir control based on conductivity and stir rpm
        stir_control = np.clip(p["conductivity_threshold"] - p["conductivity"] * 50, 0, p["conductivity_threshold"])
        stir_control = stir_control / (p["conductivity"] / 50 + 1)
        stir_control = stir_control * p["stir_threshold"]
        stir_control = np.clip(stir_control, 0, p["conductivity_threshold"])

        # Use a more sophisticated harvest control based on light umol and stir control
        if 'light_umol' in obs and 'frac' in obs:
            harvest_control = np.clip(p["light_umol"] / 1000 - stir_control * p["light_control_coefficient"], 0, p["cap"])
            harvest_control = np.clip(harvest_control, 0, p["cap"])
            harvest_control = np.clip(harvest_control, 0, 0.5)
        elif 'frac' in obs:
            harvest_control = obs['frac']
        else:
            harvest_control = 0

        frac = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"])
        frac = frac * stir_control

        return p["stir"], light_control, harvest_control


DEFAULTS = {"stir": 65.0, "light_umol": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0, "conductivity_threshold": 150, "light_control_coefficient": 0.7, "harvest_control_coefficient": 0.5}
