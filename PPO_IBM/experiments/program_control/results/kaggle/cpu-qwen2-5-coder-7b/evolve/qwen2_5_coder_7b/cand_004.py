class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest = 0
        self.current_light = p["light"]
        self.temp_target = 35.0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Adjust light intensity based on OD
        if card["od_est"] < 0.3:
            self.current_light = max(0, self.current_light - 100)  # Reduce light intensity
        elif card["od_est"] > 0.8:
            self.current_light = min(2000, self.current_light + 100)  # Increase light intensity
        
        # Adjust stirrer RPM based on temperature
        if card["temp_c"] < self.temp_target:
            stir_rpm = min(200, card["stir"] + 5)  # Increase stirrer for cooling
        elif card["temp_c"] > self.temp_target:
            stir_rpm = max(50, card["stir"] - 5)  # Decrease stirrer for heating
        
        # Harvest logic
        if card["t"] % 600 == 0 and card["t"] > self.last_harvest:
            harvest = np.mean(self.harvest_history[-600:])
            if harvest < 0.5:
                frac = 0.5  # Force harvest if average is below 50%
            else:
                frac = 0.3  # Reduce harvest fraction otherwise
            self.harvest_history.append(frac)
            self.last_harvest = card["t"]
        else:
            frac = 0.3  # Default harvest fraction
        
        return stir_rpm, self.current_light, frac
