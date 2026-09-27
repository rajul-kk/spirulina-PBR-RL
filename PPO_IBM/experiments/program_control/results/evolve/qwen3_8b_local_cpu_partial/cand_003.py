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
            "harvest_window": 600, # Number of steps to average harvest fraction
            "harvest_history": [], # Store harvest fraction history for averaging
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
        self.harvest_history.append(card["harvest_frac"])
        if len(self.harvest_history) > p["harvest_window"]:
            self.harvest_history.pop(0)
        
        # Calculate temperature control
        temp_dev = card["temp_c"] - p["temp_setpoint"]
        if abs(temp_dev) > p["temp_cap"]:
            if temp_dev > p["temp_cap"]:
                temp_control = 0.0
            else:
                temp_control = 1.0
        else:
            temp_control = p["temp_gain"] * temp_dev
        
        # Calculate OD control
        od_dev = card["od_est"] - p["setpoint"]
        if abs(od_dev) > p["od_cap"]:
            if od_dev > p["od_cap"]:
                od_control = 0.0
            else:
                od_control = 1.0
        else:
            od_control = p["od_gain"] * od_dev
        
        # Calculate light control
        light_dev = card["light_obs_umol"] - p["light"]
        if abs(light_dev) > p["light_cap"]:
            if light_dev > p["light_cap"]:
                light_control = 0.0
            else:
                light_control = 1.0
        else:
            light_control = p["light_gain"] * light_dev
        
        # Calculate stir control
        stir_dev = card["temp_c"] - p["temp_setpoint"]  # Use temperature as proxy for stir
        if abs(stir_dev) > p["stir_cap"]:
            if stir_dev > p["stir_cap"]:
                stir_control = 0.0
            else:
                stir_control = 1.0
        else:
            stir_control = p["stir_gain"] * stir_dev
        
        # Combine temperature, OD, light, and stir control to determine harvest fraction
        avg_harvest_frac = np.mean(self.harvest_history) if self.harvest_history else 0.0
        frac = float(np.clip(p["gain"] * (od_control + temp_control + light_control + stir_control), 0.0, p["cap"]))
        
        return p["stir"], p["light"], frac
