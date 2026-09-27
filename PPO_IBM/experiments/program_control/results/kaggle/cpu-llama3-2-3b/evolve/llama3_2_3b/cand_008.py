import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = {
            "turb_per_od": 250.0,
            "stir": 65.0,
            "light": 1400.0,
            "setpoint": 0.6,
            "gain": 1.0,
            "cap": 0.3,
            "temp_per_stir": 0.05,
            "temp_per_light": 0.01,
            "lux_setpoint": 30.0
        }
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        stir = np.clip(p["stir"] + p["temp_per_stir"] * (card["temp_c"] - p["setpoint"]), 50, 200)
        light = np.clip(p["light"] + p["temp_per_light"] * (card["lux"] - p["lux_setpoint"]), 0, 2000)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        return stir, light, frac

