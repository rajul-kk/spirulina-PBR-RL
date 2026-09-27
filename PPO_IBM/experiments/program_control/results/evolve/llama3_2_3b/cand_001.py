class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.g = 0.5  # growth rate gain

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        stir = np.clip(p["stir"] + p["g"] * (card["lux"] / 1400), 50, 200)  # added stir gain
        light = np.clip(p["light"] + p["g"] * (card["lux"] / 1400), 0, 2000)  # added light gain
        harvest_frac = np.clip(frac + p["g"] * (card["lux"] / 1400), 0.0, 0.5)  # added harvest gain
        return stir, light, harvest_frac
