import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = {
            "stir": 100.0,          # Higher stir for better mixing and gas exchange
            "light": 1600.0,        # Moderate light to balance growth and photo-inhibition
            "setpoint_od": 0.8,    # Target OD for harvest
            "gain": 1.2,           # Harvest gain factor
            "cap": 0.4,            # Max harvest fraction
            "turb_per_od": 250.0,  # Conversion factor for turbidity to OD
            "temp_setpoint": 35.0, # Target temperature to avoid overheating
            "temp_gain": 0.5,      # Temperature adjustment gain
            "min_stir": 50.0,      # Minimum stir RPM to avoid shear stress
            "max_stir": 150.0,     # Maximum stir RPM to prevent overheating
            "min_light": 800.0,    # Minimum light intensity to ensure growth
            "max_light": 2000.0,   # Maximum light intensity to avoid photo-inhibition
        }
        if params:
            self.p.update(params)
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.last_harvest_time = 0
        self.harvest_window = 600  # 12 hours
        self.harvest_history = []

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        od_est = card["od_est"]
        temp = card["temp_c"]
        turb = obs["turbidity_ntu"]
        hours = card["hours"]
        hours_to_harvest = card["hours_to_harvest"]
        harvests_done = card["harvests_done"]
        pump_L = card["pump_L"]
        light_obs = obs["light_umol"]  # Use the correct key
        t = obs["t"]

        # Harvest control logic
        if t % 600 == 0:  # Harvest every 600 steps (12 hours)
            if len(self.harvest_history) >= self.harvest_window:
                avg_harvest_frac = np.mean(self.harvest_history[-self.harvest_window:])
                harvest_frac = np.clip(avg_harvest_frac, 0.0, p["cap"])
            else:
                harvest_frac = 0.0
            self.harvest_history.append(harvest_frac)
            self.last_harvest_time = t
        else:
            # Calculate harvest fraction based on OD
            od_diff = od_est / p["setpoint_od"]
            harvest_frac = np.clip(p["gain"] * (od_diff - 1.0), 0.0, p["cap"])
            self.harvest_history.append(harvest_frac)
        
        # Temperature control
        temp_diff = temp - p["temp_setpoint"]
        temp_adjust = np.clip(p["temp_gain"] * temp_diff, -100.0, 100.0)
        
        # Adjust light intensity to avoid overheating
        light = np.clip(p["light"] + temp_adjust, p["min_light"], p["max_light"])
        
        # Adjust stir speed based on temperature
        stir = np.clip(p["stir"] + temp_adjust, p["min_stir"], p["max_stir"])
        
        return stir, light, harvest_frac
