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
        if 'lux' in obs and 'light_umol' in obs:
            light_control = np.clip(np.max([p["light_umol"], 1000 - obs["lux"] / 30]), 0, p["light_umol"])
            # Use a light control strategy that reduces light when it's high to prevent overheating
            if obs["lux"] > 2000:
                light_control = max(0, light_control - (obs["lux"] - 2000) / 500)
        else:
            light_control = p["light_umol"]
        
        # Use a more sophisticated stir control based on conductivity and stir rpm
        stir_control = np.clip(p["conductivity_threshold"] * p["conductivity"] / 50, 0, p["conductivity_threshold"])
        stir_control = stir_control / (p["conductivity"] / 50 + 1)
        stir_control = stir_control * p["stir"]
        # Use a stir control strategy that reduces stir when it's high to prevent shear and heat
        if p["conductivity"] > 1000:
            stir_control = max(0, stir_control - (p["conductivity"] - 1000) / 200)

        frac = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"])
        frac = frac * stir_control

        return p["stir"], light_control, frac


DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0, "conductivity_threshold": 150, "light_umol_threshold": 1200,
            "stir_threshold": 150, "stir_control_coefficient": 0.8, "light_control_coefficient": 0.7,
            "light_heating_threshold": 2200, "shear_heating_threshold": 1100}
