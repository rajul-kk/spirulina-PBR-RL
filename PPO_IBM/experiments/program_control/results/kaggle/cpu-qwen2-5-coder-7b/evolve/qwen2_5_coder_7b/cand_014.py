class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest = 0
        self.max_growth_rate = 0.0
        self.current_growth_rate = 0.0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Update growth rate
        if card["hours"] > 0:
            self.current_growth_rate = card["growth_per_h"]
            if self.current_growth_rate > self.max_growth_rate:
                self.max_growth_rate = self.current_growth_rate
        
        # Light intensity adjustment based on OD and growth rate
        if card["od_est"] < 0.3:
            light = 500.0
        elif card["od_est"] > 1.5:
            light = 2000.0
        else:
            light = 1000.0 + (card["od_est"] - 0.3) * 500.0
        
        # Harvest logic
        if card["t"] % 600 == 0 and card["t"] > self.last_harvest:
            harvest = np.mean(self.harvest_history[-600:])
            if harvest < 0.5:
                frac = 0.5  # Force harvest if average is below 50%
            else:
                frac = 0.3  # Reduce harvest fraction otherwise
            self.harvest_history.append(frac)
            self.last_harvest = card["t"]
        
        # Stirrer adjustment based on growth rate
        stir = 100.0 + (self.current_growth_rate / self.max_growth_rate) * 100.0
        
        return min(stir, 200), light, frac
