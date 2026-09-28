class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest = 0
        self.max_growth_rate = 0.0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        
        # Update max growth rate
        if card["growth_per_h"] > self.max_growth_rate:
            self.max_growth_rate = card["growth_per_h"]
        
        # Harvest logic
        if card["t"] % 600 == 0 and card["t"] > self.last_harvest:
            harvest = np.mean(self.harvest_history[-600:])
            if harvest < 0.5:
                frac = 0.5  # Force harvest if average is below 50%
            else:
                frac = 0.3  # Reduce harvest fraction otherwise
            self.harvest_history.append(frac)
            self.last_harvest = card["t"]
        
        # Adjust light intensity based on OD
        if card["od_est"] < 0.3:
            p["light"] = max(0, p["light"] - 100)  # Reduce light intensity
        else:
            p["light"] = min(2000, p["light"] + 100)  # Increase light intensity
        
        # Adjust stir RPM based on growth rate
        if card["growth_per_h"] < 0.1 * self.max_growth_rate:
            p["stir"] = max(50, p["stir"] - 5)  # Reduce stir RPM
        else:
            p["stir"] = min(200, p["stir"] + 5)  # Increase stir RPM
        
        return p["stir"], p["light"], frac
