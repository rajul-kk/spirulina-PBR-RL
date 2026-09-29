import numpy as np
from collections import deque
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = dict({"stir": 150.0, "light": 1800.0, "setpoint": 0.75, "gain": 0.5, "cap": 0.5}, **(params or {}))
        self.harvest_history = deque(maxlen=600)
        self.last_harvest_time = -600
        self.max_growth_rate = 0.5  # Placeholder for dynamic growth rate

    def act(self, obs):
        p = self.p
        
        # Update harvest history
        if obs["t"] % 600 == 0:
            self.harvest_history.append(obs["pump_L"])
            self.last_harvest_time = obs["t"]
        
        # Calculate the target OD
        target_od = 0.75
        
        # Calculate the OD estimation from turbidity
        od_est = obs["turbidity_ntu"] / 250.0
        
        # Adjust light intensity based on OD estimation and mean harvest fraction
        if od_est < target_od:
            light_adjustment = 1.0
        else:
            light_adjustment = max(0, 1 - (obs["t"] - self.last_harvest_time) / 600)
        
        light = p["light"] * light_adjustment
        
        # Ensure light stays within bounds
        light = max(0, min(2000, light))
        
        # Adjust stir rate based on temperature and OD
        if obs["temp_c"] > 34:
            stir_adjustment = 1.0
        else:
            stir_adjustment = max(0, (35 - obs["temp_c"]) / 5)
        
        stir = p["stir"] * stir_adjustment
        
        # Ensure stir rate stays within bounds
        stir = max(50, min(200, stir))
        
        # Adjust harvest fraction based on OD and target
        frac = float(np.clip(p["gain"] * (od_est / target_od - 1.0), 0.0, p["cap"]))
        
        return stir, light, frac

