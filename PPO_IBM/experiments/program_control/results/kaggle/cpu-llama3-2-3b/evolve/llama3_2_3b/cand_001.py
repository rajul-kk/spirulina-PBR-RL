class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        stir = np.clip(p["stir"] * np.tanh(card["temp_c"] - p["temp_setpoint"]), 50, 200)  # added temperature dependence
        light = np.clip(p["light"] * np.tanh(card["lux"] - p["lux_setpoint"]), 0, 2000)  # added light dependence
        return stir, light, frac
