import numpy as np

class Controller:
    def __init__(self, params=None):
        self.p = {
            "stir": 65.0,
            "light": 1400.0,
            "setpoint": 0.6,
            "gain": 1.0,
            "cap": 0.3,
            "turb_per_od": 250.0,
            "temp_setpoint": 35.0,
            "lux_setpoint": 30.0,
        }
        self.p = {**self.p, **(params or {})}
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        stir = np.clip(p["stir"] * np.tanh(card["temp_c"] - p["temp_setpoint"]), 50, 200)
        light = np.clip(p["light"] * np.tanh(card["lux"] - p["lux_setpoint"]), 0, 2000)
        if card["turb"] < 100:  # Harvest when turbidity is low
            harvest_frac = 0.5
        else:
            harvest_frac = 0.0
        return stir, light, harvest_frac

