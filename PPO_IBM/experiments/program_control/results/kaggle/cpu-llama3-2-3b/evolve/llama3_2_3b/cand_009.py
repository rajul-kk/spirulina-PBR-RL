class Controller:
    def __init__(self, params=None):
        self.p = params or {}
        self.defaults = {
            "stir": 65.0,  # default stir speed
            "light": 1400.0,  # default light intensity
            "setpoint": 0.6,  # default growth rate setpoint
            "gain": 1.0,  # default gain for harvest fraction
            "cap": 0.30,  # default harvest fraction cap
            "turb_per_od": 250.0,  # default turbidity per OD unit
            "temp_setpoint": 35.0,  # default temperature setpoint
            "lux_setpoint": 30.0  # default light intensity setpoint
        }
        self.p = {k: v for k, v in self.defaults.items() if k in self.p or v != self.defaults[k]}
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        stir = np.clip(p["stir"] * np.tanh(card["temp_c"] - p["temp_setpoint"]), 50, 200)  # added temperature dependence
        light = np.clip(p["light"] * np.tanh(card["lux"] - p["lux_setpoint"]), 0, 2000)  # added light dependence
        return stir, light, frac
