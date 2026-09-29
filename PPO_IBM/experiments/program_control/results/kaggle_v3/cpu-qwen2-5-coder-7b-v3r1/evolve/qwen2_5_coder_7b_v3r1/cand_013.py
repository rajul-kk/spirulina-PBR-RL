import numpy as np
from process_state import ProcessState

DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.7, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0}

class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest_time = 0
        self.last_growth_rate = 0
        self.last_temp = 0
        self.last_ph = 0
        self.last_od = 0

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
        
        # Adjust light and stir based on temperature, growth rate, pH, and OD
        if card["temp_c"] > 34.5:
            p["light"] = min(p["light"] + 50, 2000)
            p["stir"] = min(p["stir"] + 5, 200)
        elif card["temp_c"] < 33.5:
            p["light"] = max(p["light"] - 50, 0)
            p["stir"] = max(p["stir"] - 5, 50)
        
        if card["ph"] > 9.9:
            p["light"] = min(p["light"] - 50, 2000)
        elif card["ph"] < 9.5:
            p["light"] = min(p["light"] + 50, 2000)
        
        if card["od_est"] > self.last_od:
            frac = min(frac + 0.1, 0.5)
        elif card["od_est"] < self.last_od:
            frac = max(frac - 0.1, 0.0)
        
        # Ensure the harvest fraction does not exceed 0.5
        frac = min(frac, 0.5)
        
        # Update last growth rate, temperature, pH, and OD for next step
        self.last_growth_rate = card["growth_per_h"]
        self.last_temp = card["temp_c"]
        self.last_ph = card["ph"]
        self.last_od = card["od_est"]
        
        return p["stir"], p["light"], frac
