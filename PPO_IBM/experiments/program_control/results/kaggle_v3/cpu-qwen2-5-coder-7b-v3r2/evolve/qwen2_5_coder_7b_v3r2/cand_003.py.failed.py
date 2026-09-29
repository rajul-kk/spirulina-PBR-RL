import numpy as np
from collections import deque

class Controller:
    def __init__(self, params=None):
        self.p = dict({"stir": 100.0, "light": 1200.0, "setpoint": 0.7, "gain": 0.5, "cap": 0.5}, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = deque(maxlen=600)
        self.last_harvest_time = -600

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Update harvest history
        if card["t"] % 600 == 0:
            self.harvest_history.append(card["pump_L"])
            self.last_harvest_time = card["t"]
        
        # Calculate the target OD
        target_od = 0.75
        
        # Calculate the OD estimation from turbidity
        od_est = card["od_est"]
        
        # Calculate the harvest fraction based on OD and target
        frac = float(np.clip(p["gain"] * (od_est / target_od - 1.0), 0.0, p["cap"]))
        
        # Adjust light intensity based on OD estimation and mean harvest fraction
        if od_est < target_od:
            light_adjustment = 1.0
        else:
            light_adjustment = max(0, 1 - (card["t"] - self.last_harvest_time) / 600)
        
        light = p["light"] * light_adjustment
        
        # Ensure light stays within bounds
        light = max(0, min(2000, light))
        
        # Adjust stir rate based on temperature
        if card["temp_c"] > 34:
            stir_adjustment = 1.0
        else:
            stir_adjustment = max(0, (35 - card["temp_c"]) / 5)
        
        stir = p["stir"] * stir_adjustment
        
        # Ensure stir rate stays within bounds
        stir = max(50, min(200, stir))
        
        return stir, light, frac
