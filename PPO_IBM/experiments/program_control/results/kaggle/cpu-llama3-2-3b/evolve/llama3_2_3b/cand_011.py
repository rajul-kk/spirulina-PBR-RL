class Controller:
    def __init__(self, params=None):
        self.p = {
            "stir": 150.0,  # Stir rate range: 50-200 RPM
            "light": 1500.0,  # Light intensity range: 0-2000 umol/m2/s
            "temp_setpoint": 35.0,  # Setpoint temperature range: 30-40°C
            "gain": 1.0,  # Gain for OD control
            "cap": 0.3,  # Harvest fraction cap
            "turb_per_od": 250.0
        }
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["temp_setpoint"]), 0.0, p["cap"]))
        stir = np.clip(p["stir"] * np.tanh((card["temp_c"] - p["temp_setpoint"]) / 5.0), 50, 200)
        light = np.clip(p["light"] * np.tanh((card["lux"] - p["lux_setpoint"]) / 500.0), 0, 2000)
        return stir, light, frac
