import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"], ph=self.p["pH_bias"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        # Use more sophisticated stir control based on conductivity and stir rpm
        stir_control = np.clip(p["conductivity_threshold"] * card["conductivity"] / 50, 0, p["conductivity_threshold"])
        stir_control = stir_control * p["stir_control_coefficient"]

        # Use a more sophisticated light control based on lux and light umol
        light_control = min(p["light_umol"], 1000 - card["lux"] / 30)
        light_control = np.clip(light_control, 0, p["light_umol"])

        # Calculate harvest fraction based on light intensity and growth rate
        growth_rate = card["growth_per_h"]
        frac = np.clip(p["gain"] * (growth_rate / p["setpoint"] - 1.0), 0.0, p["cap"])
        frac = frac * stir_control * light_control

        return p["stir"], light_control, frac

DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0, "conductivity_threshold": 150, "light_umol_threshold": 1200,
            "stir_threshold": 150, "stir_control_coefficient": 0.8, "light_control_coefficient": 0.7,
            "pH_bias": -0.1, "growth_rate_offset": 0.2}
