import numpy as np

class Controller:
    def __init__(self, params=None):
        self.p = {
            "turb_per_od": 250.0,
            "gain": 1.0,
            "cap": 0.3,
            "temp_setpoint": 35.0,
            "stir_min": 50.0,
            "stir_max": 200.0,
            "light_min": 0.0,
            "light_max": 2000.0,
            "light_setpoint": 30.0,
        }
        self.p.update(params or {})

    def act(self, obs):
        p = self.p
        card = self._update_obs(obs)
        stir = np.clip(p["stir_min"] + (p["stir_max"] - p["stir_min"]) * np.tanh(card["temp_c"] - p["temp_setpoint"]), p["stir_min"], p["stir_max"])
        light = np.clip(p["light_min"] + (p["light_max"] - p["light_min"]) * np.tanh(card["lux"] - p["light_setpoint"]), p["light_min"], p["light_max"])
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["turb_per_od"] - 1.0), 0.0, p["cap"]))
        return stir, light, frac

    def _update_obs(self, obs):
        card = ProcessState(turb_per_od=self.p["turb_per_od"])
        card.update(obs)
        return card
