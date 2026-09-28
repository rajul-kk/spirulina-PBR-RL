class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest = 0
        self.max_growth_rate = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Update max growth rate
        if card["growth_per_h"] > self.max_growth_rate:
            self.max_growth_rate = card["growth_per_h"]
        
        # Calculate target OD based on max growth rate
        target_od = self.max_growth_rate * card["dt_h"]
        
        # Adjust light intensity based on OD
        if card["od_est"] < target_od / 2:
            light = 0.5 * p["light"]
        else:
            light = p["light"]
        
        # Harvest logic
        if card["t"] % 600 == 0 and card["t"] > self.last_harvest:
            harvest = np.mean(self.harvest_history[-600:])
            if harvest < 0.5:
                frac = 0.5  # Force harvest if average is below 50%
            else:
                frac = 0.3  # Reduce harvest fraction otherwise
            self.harvest_history.append(frac)
            self.last_harvest = card["t"]
        
        return p["stir"], light, frac

