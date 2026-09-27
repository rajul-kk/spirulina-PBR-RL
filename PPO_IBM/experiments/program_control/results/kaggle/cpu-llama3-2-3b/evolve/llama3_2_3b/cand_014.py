class Controller:
    def __init__(self, params=None):
        self.p = {
            "turb_per_od": 250.0,
            "gain": 1.0,
            "setpoint": 0.75,
            "cap": 0.3,
            "stir_min": 50,
            "stir_max": 200,
            "light_min": 0,
            "light_max": 2000,
            "temp_setpoint": 35.0,
            "temp_min": 30.0,
            "temp_max": 40.0,
        }
        self.p = {k: v for k, v in self.p.items() if v != DEFAULTS[k]}
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        stir = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0) * (p["temp_c"] - p["temp_setpoint"]) / (p["temp_max"] - p["temp_min"]),
                     p["stir_min"], p["stir_max"])
        light = np.clip(p["gain"] * (card["lux"] - p["lux_setpoint"]) / (p["light_max"] - p["light_min"]),
                        p["light_min"], p["light_max"])
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        return stir, light, frac

