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
        
        # Update max growth rate observed
        if card["growth_per_h"] > self.max_growth_rate:
            self.max_growth_rate = card["growth_per_h"]
        
        # Adjust light intensity based on OD
        if card["od_est"] < 0.3:
            light = 1000.0  # Reduce light intensity to prevent culture from dying out
        elif card["od_est"] > 0.7:
            light = 2000.0  # Increase light intensity for better growth
        else:
            light = 1400.0  # Optimal light intensity
        
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
