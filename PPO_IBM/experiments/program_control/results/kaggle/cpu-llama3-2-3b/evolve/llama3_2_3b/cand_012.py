import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.params = {
            "turb_per_od": 250.0,
            "stir_min": 50.0,
            "stir_max": 200.0,
            "light_min": 0.0,
            "light_max": 2000.0,
            "gain": 1.0,
            "setpoint": 0.75,
            "cap": 0.3,
            "temp_setpoint": 35.0
        }
        self.p = self.params.copy()
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        stir = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), p["stir_min"], p["stir_max"])
        light = np.clip(p["gain"] * (card["lux"] - p["lux_setpoint"]), p["light_min"], p["light_max"])
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        return stir, light, frac
