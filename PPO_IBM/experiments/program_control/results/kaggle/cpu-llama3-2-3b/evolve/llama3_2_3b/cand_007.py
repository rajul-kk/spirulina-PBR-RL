class Controller:
    def __init__(self, params=None):
        self.p = params or {}
        self.p["turb_per_od"] = 250.0
        self.p["gain"] = 1.0
        self.p["setpoint"] = 0.75
        self.p["cap"] = 0.3
        self.p["stir"] = 65.0
        self.p["light"] = 1400.0
        self.p["temp_setpoint"] = 35.0
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        stir = np.clip(p["stir"] * np.tanh(card["temp_c"] - p["temp_setpoint"]), 50, 200)
        light = np.clip(p["light"] * np.tanh(card["lux"] - p["lux_setpoint"]), 0, 2000)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        return stir, light, frac

