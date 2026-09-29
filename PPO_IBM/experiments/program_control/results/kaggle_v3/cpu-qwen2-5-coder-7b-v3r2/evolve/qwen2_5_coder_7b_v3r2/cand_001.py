"""The demo expert's proportional-harvest law on sensors only: OD is estimated from turbidity.
DEFAULTS are the hand-tuned values; cmaes_tune.py searches over exactly these keys."""
import numpy as np

from process_state import ProcessState

DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0}

class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Update harvest history
        if card["t"] % 600 == 0:
            self.harvest_history.append(card["pump_L"])
            if len(self.harvest_history) > 600:
                self.harvest_history.pop(0)
        
        # Calculate the mean harvest fraction for the last 600 steps
        mean_harvest_frac = np.mean([max(0, min(1, frac)) for frac in self.harvest_history])
        
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
            light_adjustment = max(0, 1 - mean_harvest_frac)
        
        light = p["light"] * light_adjustment
        
        # Ensure light stays within bounds
        light = max(0, min(2000, light))
        
        return p["stir"], light, frac
