class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.t = 0
        self.harvested_mg = 0
        self.harvest_frac = np.zeros((7200))

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip((card["od_est"] - p["setpoint"]) / (card["od_est"] + 1), 0.0, p["cap"]))
        self.harvest_frac[self.t] = frac
        if self.t % 600 == 0:
            self.harvested_mg = np.mean(self.harvest_frac[-600:])
            self.t = 0
        stir = np.clip(p["stir"] + self.t * 1, 50, 200)
        light = np.clip(p["light"] + self.t * 10, 0, 2000)
        return stir, light, self.harvest_frac[-600:].mean()

