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
        
        # Update max growth rate
        if card["growth_per_h"] > self.max_growth_rate:
            self.max_growth_rate = card["growth_per_h"]
        
        # Calculate fraction to harvest
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
        
        # Adjust light intensity based on OD
        if card["od_est"] < 0.3:
            p["light"] = max(0, p["light"] - 200)  # Reduce light intensity
        else:
            p["light"] = min(2000, p["light"] + 200)  # Increase light intensity
        
        # Adjust stir rate based on temperature
        if card["temp_c"] > 38:
            p["stir"] = min(200, p["stir"] + 10)  # Increase stir rate
        elif card["temp_c"] < 34:
            p["stir"] = max(50, p["stir"] - 10)  # Decrease stir rate
        
        return p["stir"], p["light"], frac
