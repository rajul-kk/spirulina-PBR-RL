```python
import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        # Default parameters tuned for optimal biomass harvest
        self.p = {
            "stir": 120.0,        # Higher mixing for better gas exchange
            "light": 1600.0,      # Moderate light to avoid overheating
            "setpoint": 0.6,      # Target OD level
            "gain": 1.5,          # Gain for harvest fraction
            "cap": 0.4,           # Max harvest fraction
            "turb_per_od": 250.0, # Conversion factor for turbidity to OD
            "temp_setpoint": 35.0, # Target temperature to avoid overheating
            "temp_gain": 0.05,    # Gain for temperature control
            "temp_cap": 1.0,      # Max temperature deviation allowed
            "od_gain": 1.0,       # Gain for OD control
            "od_cap": 1.0,        # Max OD deviation allowed
            "light_gain": 0.05,   # Gain for light control
            "light_cap": 1.0,     # Max light deviation allowed
            "stir_gain": 0.05,    # Gain for stir control
            "stir_cap": 1.0,      # Max stir deviation allowed
            "harvest_window": 600, # Number of steps to consider for harvest fraction
            "harvest_history": [], # History of harvest fractions
        }
        # Update with user parameters if provided
        if params:
            self.p.update(params)
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.temp_history = []
        self.od_history = []
        self.light_history = []
        self.stir_history = []
        self.harvest_history = []

    def act(self, obs):
        card = self.ps.update(obs)
        p = self.p
        
        # Update harvest history
        self.harvest_history.append(self.harvest_frac)
        if len(self.harvest_history) > p["harvest_window"]:
            self.harvest_history.pop(0)
        
        # Temperature control
        temp_dev = card["temp_c"] - p["temp_setpoint"]
        if abs(temp_dev) > p["temp_cap"]:
            temp_control = 1.0 if temp_dev > p["temp_cap"] else 0.0
        else:
            temp_control = p["temp_gain"] * temp_dev
        
        # OD control
        od_dev = card["od_est"] - p["setpoint"]
        if abs(od_dev) > p["od_cap"]:
            od_control = 1.0 if od_dev > p["od_cap"] else 0.0
        else:
            od_control = p["od_gain"] * od_dev
        
        # Light control
        light_dev = card["light_obs_umol"] - p["light"]
        if abs(light_dev) > p["light_cap"]:
            light_control = 1.0 if light_dev > p["light_cap"] else 0.0
        else:
            light_control = p["light_gain"] * light_dev
        
        # Stir control (based on temperature)
        stir_dev = card["temp_c"] - p["temp_setpoint"]
        if abs(stir_dev) > p["stir_cap"]:
            stir_control = 1.0 if stir_dev > p["stir_cap"] else 0.0
        else:
            stir_control = p["stir_gain"] * stir_dev
        
        # Calculate harvest fraction using moving average of harvest history
        if len(self.harvest_history) >= p["harvest_window"]:
            avg_frac = np.mean(self.harvest_history)
        else:
            avg_frac = 0.0
        
        # Combine controls to determine harvest fraction
        frac = float(np.clip(p["gain"] * (od_control + temp_control + light_control + stir_control), 0.0, p["cap"]))
        
        # Use a combination of calculated fraction and moving average to smooth harvest
        self.harvest_frac = np.clip(avg_frac + frac, 0.0, p["cap"])
        
        return p["stir"], p["light"], self.harvest_frac
```