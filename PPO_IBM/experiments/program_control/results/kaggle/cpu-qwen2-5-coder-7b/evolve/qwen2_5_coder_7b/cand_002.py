class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        
        # Harvest logic
        if card["t"] % 600 == 0 and card["t"] > self.last_harvest:
            harvest = np.mean(self.harvest_history[-600:])
            if harvest < 0.5:
                frac = 0.5  # Force harvest if average is below 50%
            else:
                frac = 0.3  # Reduce harvest fraction otherwise
            self.harvest_history.append(frac)
            self.last_harvest = card["t"]
        
        return p["stir"], p["light"], frac

