import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = {
            "stir": 65.0,          # Optimal stir speed for gas exchange and shear balance
            "light": 1400.0,       # Moderate light intensity to balance growth and photo-inhibition
            "setpoint_od": 0.75,   # Target OD for harvest
            "gain": 1.0,           # Harvest gain factor
            "cap": 0.3,            # Max harvest fraction
            "turb_per_od": 250.0,  # Conversion factor for turbidity to OD
            "temp_setpoint": 35.0, # Target temperature to avoid overheating
            "temp_gain": 0.3,      # Temperature adjustment gain
            "min_stir": 50.0,      # Minimum stir RPM to avoid shear stress
            "max_stir": 150.0,     # Maximum stir RPM to prevent overheating
            "min_light": 800.0,    # Minimum light intensity to ensure growth
            "max_light": 2000.0,   # Maximum light intensity to avoid photo-inhibition
            "harvest_window": 600, # 12 hours for harvest averaging
            "od_gain": 1.2,        # Additional gain for OD control
            "temp_decay": 0.98,    # Decay factor for temperature adjustment
            "harvest_delay": 120,  # Delay harvest for better growth
            "harvest_threshold": 1.0,  # OD threshold for harvest
            "harvest_delay_counter": 0, # Counter for harvest delay
        }
        if params:
            self.p.update(params)
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.temp_adjust = 0.0
        self.harvest_delay_counter = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        od_est = card["od_est"]
        temp = card["temp_c"]
        light_obs = obs.get("light_umol", 0.0)  # Safeguard against missing key
        t = obs["t"]
        hours = card["hours"]
        hours_to_harvest = card["hours_to_harvest"]
        harvests_done = card["harvests_done"]
        pump_L = card["pump_L"]
        light_obs = np.clip(light_obs, p["min_light"], p["max_light"])

        # Temperature control with decay
        temp_diff = temp - p["temp_setpoint"]
        temp_adjust = np.clip(p["temp_gain"] * temp_diff, -100.0, 100.0)
        self.temp_adjust = p["temp_decay"] * self.temp_adjust + temp_adjust

        # Harvest control logic
        if t % p["harvest_window"] == 0:  # Harvest every 600 steps (12 hours)
            if len(self.harvest_history) >= p["harvest_window"]:
                avg_harvest_frac = np.mean(self.harvest_history[-p["harvest_window"]:])
                harvest_frac = np.clip(avg_harvest_frac, 0.0, p["cap"])
            else:
                harvest_frac = 0.0
            self.harvest_history.append(harvest_frac)
        else:
            # Calculate harvest fraction based on OD with added gain and delay
            if self.harvest_delay_counter > p["harvest_delay"]:
                od_diff = od_est / p["setpoint_od"]
                harvest_frac = np.clip(p["gain"] * (od_diff - 1.0) + p["od_gain"] * (od_diff - 1.0), 0.0, p["cap"])
                self.harvest_delay_counter = 0
            else:
                harvest_frac = 0.0
                self.harvest_delay_counter += 1
            self.harvest_history.append(harvest_frac)
        
        # Adjust light intensity to avoid overheating
        light = np.clip(p["light"] + self.temp_adjust, p["min_light"], p["max_light"])
        
        # Adjust stir speed based on temperature
        stir = np.clip(p["stir"] + self.temp_adjust, p["min_stir"], p["max_stir"])
        
        return stir, light, harvest_frac
