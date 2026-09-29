import numpy as np
from process_state import ProcessState

DEFAULTS = {"stir": 100.0, "light": 1000.0, "setpoint": 0.75, "gain": 1.0, "cap": 0.5,
            "turb_per_od": 250.0}

class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.prev_od_est = None
        self.prev_temp_c = None
        self.harvest_history = []

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Adjust light based on current OD and temperature
        light_factor = max(0.1, min(1.0, 1.5 - (card["od_est"] / 1.5)))
        light = p["light"] * light_factor
        
        # Adjust stir based on current temperature
        stir_factor = max(50, min(200, 100 + (card["temp_c"] - 35) * 10))
        stir = stir_factor
        
        # Adjust harvest fraction based on previous and current OD estimates
        if self.prev_od_est is not None:
            growth_rate = (card["od_est"] - self.prev_od_est) / p["dt_h"]
            harvest_frac = min(0.5, max(0.0, (growth_rate - 0.1) / 0.2))
        else:
            harvest_frac = 0.0
        
        # Update harvest history and calculate average harvest fraction for the last 600 steps
        self.harvest_history.append(harvest_frac)
        if len(self.harvest_history) > 600:
            self.harvest_history.pop(0)
        avg_harvest_frac = sum(self.harvest_history) / len(self.harvest_history)
        
        # If the average harvest fraction is less than 0.25, decrease the harvest fraction
        if avg_harvest_frac < 0.25:
            harvest_frac = 0.25
        
        self.prev_od_est = card["od_est"]
        self.prev_temp_c = card["temp_c"]
        
        return stir, light, harvest_frac
