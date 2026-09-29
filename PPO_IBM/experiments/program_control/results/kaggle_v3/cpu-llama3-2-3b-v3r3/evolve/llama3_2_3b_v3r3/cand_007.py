class Controller:
    DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.3,
                "turb_per_od": 250.0}

    def __init__(self, params=None): 
        self.p = dict(self.DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"], growth_per_h=self.p["gain"], temp_c=self.p["setpoint"])
        self.t = 0
        self.harvest_frac = np.zeros((7200))

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip((card["od_est"] - p["setpoint"]) / (card["od_est"] + 1), 0.0, p["cap"]))
        stir = np.clip(p["stir"] + self.t * 0.5, 50, 200)
        light = np.clip(p["light"] + self.t * 5, 0, 2000)
        return stir, light, frac
