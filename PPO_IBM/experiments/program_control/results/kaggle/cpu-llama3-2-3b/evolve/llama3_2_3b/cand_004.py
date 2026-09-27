class Controller:
    def __init__(self, params=None):
        self.p = params or {}
        self.p["default_stir"] = 65.0
        self.p["default_light"] = 1400.0
        self.p["default_frac"] = 0.6
        self.p["gain"] = 1.0
        self.p["turb_per_od"] = 250.0

    def act(self, obs):
        p = self.p
        card = ProcessState(turb_per_od=p["turb_per_od"]).update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["default_frac"] - 1.0), 0.0, p["default_frac"]))
        stir = np.clip(p["default_stir"] * np.tanh(card["temp_c"] - p["default_temp_c"]), 50, 200)
        light = np.clip(p["default_light"] * np.tanh(card["lux"] - p["default_light"]), 0, 2000)
        return stir, light, frac

