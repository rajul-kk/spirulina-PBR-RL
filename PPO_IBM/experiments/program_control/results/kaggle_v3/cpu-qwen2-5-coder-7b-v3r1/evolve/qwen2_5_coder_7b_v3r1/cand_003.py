import numpy as np
from process_state import ProcessState

DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0}

class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest_time = 0
        self.last_growth_rate = 0
        self.last_temp = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Update harvest history
        if card["hours"] % 600 == 0:
            self.harvest_history.append(card["pump_L"])
            self.last_harvest_time = card["hours"]
        
        # Calculate average harvest over the last 600 steps
        avg_harvest = np.mean(self.harvest_history[-600:]) if self.harvest_history else 0
        
        # Proportional-harvest law
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        
        # Adjust light and stir based on temperature and growth rate
        if card["temp_c"] > 34.5:
            p["light"] = min(p["light"] + 100, 2000)
            p["stir"] = min(p["stir"] + 5, 200)
        elif card["temp_c"] < 33.5:
            p["light"] = max(p["light"] - 100, 0)
            p["stir"] = max(p["stir"] - 5, 50)
        
        # Adjust harvest fraction based on average harvest, growth rate, and temperature
        if avg_harvest > 20000 and card["growth_per_h"] > self.last_growth_rate and card["temp_c"] > self.last_temp:
            frac = min(frac + 0.1, 0.5)
        elif avg_harvest < 10000 and card["growth_per_h"] < self.last_growth_rate and card["temp_c"] < self.last_temp:
            frac = max(frac - 0.1, 0.0)
        
        # Ensure the harvest fraction does not exceed 0.5
        frac = min(frac, 0.5)
        
        # Update last growth rate and temperature for next step
        self.last_growth_rate = card["growth_per_h"]
        self.last_temp = card["temp_c"]
        
        return p["stir"], p["light"], frac

