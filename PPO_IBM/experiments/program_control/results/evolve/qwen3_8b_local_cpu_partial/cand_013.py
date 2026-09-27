import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        # Default parameters tuned for optimal biomass harvest
        self.p = {
            "stir": 120.0,        # Higher mixing for better gas exchange
            "light": 1400.0,      # Moderate light to avoid overheating
            "setpoint": 0.6,      # Target OD level
            "gain": 1.2,          # Gain for harvest fraction
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
            "harvest_history": [0.0] * 600,  # 600-step history for harvest fraction
            "harvest_window_size": 600,
            "temp_history": [0.0] * 600,  # History for temperature
            "od_history": [0.0] * 600,    # History for OD
            "light_history": [0.0] * 600, # History for light
            "stir_history": [0.0] * 600,  # History for stir
            "temp_smooth": 0.0,           # Smoothed temperature
            "od_smooth": 0.0,             # Smoothed OD
            "light_smooth": 0.0,          # Smoothed light
            "stir_smooth": 0.0,           # Smoothed stir
        }
        # Update with user parameters if provided
        if params:
            self.p.update(params)
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.temp_history = [0.0] * self.p["harvest_window_size"]
        self.od_history = [0.0] * self.p["harvest_window_size"]
        self.light_history = [0.0] * self.p["harvest_window_size"]
        self.stir_history = [0.0] * self.p["harvest_window_size"]
        self.harvest_history = [0.0] * self.p["harvest_window_size"]

    def act(self, obs):
        card = self.ps.update(obs)
        p = self.p
        
        # Update history arrays
        self.temp_history.pop(0)
        self.temp_history.append(card["temp_c"])
        self.od_history.pop(0)
        self.od_history.append(card["od_est"])
        self.light_history.pop(0)
        self.light_history.append(card["light_obs_umol"])
        self.stir_history.pop(0)
        self.stir_history.append(card["stir"])
        self.harvest_history.pop(0)
        self.harvest_history.append(0.0)  # Default to 0 unless we compute it
        
        # Smooth temperature, OD, light, and stir
        self.p["temp_smooth"] = np.mean(self.temp_history)
        self.p["od_smooth"] = np.mean(self.od_history)
        self.p["light_smooth"] = np.mean(self.light_history)
        self.p["stir_smooth"] = np.mean(self.stir_history)
        
        # Calculate temperature control
        temp_dev = self.p["temp_smooth"] - p["temp_setpoint"]
        if abs(temp_dev) > p["temp_cap"]:
            if temp_dev > p["temp_cap"]:
                temp_control = 0.0
            else:
                temp_control = 1.0
        else:
            temp_control = p["temp_gain"] * temp_dev
        
        # Calculate OD control
        od_dev = self.p["od_smooth"] - p["setpoint"]
        if abs(od_dev) > p["od_cap"]:
            if od_dev > p["od_cap"]:
                od_control = 0.0
            else:
                od_control = 1.0
        else:
            od_control = p["od_gain"] * od_dev
        
        # Calculate light control
        light_dev = self.p["light_smooth"] - p["light"]
        if abs(light_dev) > p["light_cap"]:
            if light_dev > p["light_cap"]:
                light_control = 0.0
            else:
                light_control = 1.0
        else:
            light_control = p["light_gain"] * light_dev
        
        # Calculate stir control
        stir_dev = self.p["temp_smooth"] - p["temp_setpoint"]  # Use temperature as proxy for stir
        if abs(stir_dev) > p["stir_cap"]:
            if stir_dev > p["stir_cap"]:
                stir_control = 0.0
            else:
                stir_control = 1.0
        else:
            stir_control = p["stir_gain"] * stir_dev
        
        # Smooth harvest fraction control using history
        harvest_avg = np.mean(self.harvest_history)
        harvest_control = p["gain"] * (od_control + temp_control + light_control + stir_control + harvest_avg)
        
        # Calculate final harvest fraction
        frac = float(np.clip(harvest_control, 0.0, p["cap"]))
        
        # Update harvest history
        self.harvest_history.pop(0)
        self.harvest_history.append(frac)
        
        return p["stir"], p["light"], frac
