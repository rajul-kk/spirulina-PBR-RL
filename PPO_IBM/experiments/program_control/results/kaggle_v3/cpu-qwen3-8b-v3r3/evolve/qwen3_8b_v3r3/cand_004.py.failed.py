import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = {
            "stir": 100.0,       # Optimal mixing for gas exchange without excessive shear
            "light": 1600.0,     # High light for growth but with saturation and heat concerns
            "setpoint": 0.7,     # Target OD for harvest fraction control
            "gain": 1.2,         # Sensitivity of harvest fraction to OD deviation
            "cap": 0.45,         # Maximum harvest fraction (0.5 is the cap)
            "turb_per_od": 250.0,  # Conversion factor between turbidity and OD
            "temp_gain": 0.05,   # Temperature adjustment factor for light
            "temp_cap": 38.0,    # Max temperature to avoid overheating
            "od_gain": 1.5,      # OD adjustment factor for light
            "od_cap": 1.2,       # Max OD to avoid saturation
        }
        if params:
            self.p.update(params)
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.temp_buffer = []
        self.od_buffer = []
        self.light_buffer = []
        self.harvest_buffer = []
        self.last_light = self.p["light"]
        self.last_stir = self.p["stir"]
        self.last_frac = 0.0

    def act(self, obs):
        card = self.ps.update(obs)
        turbidity = obs["turbidity_ntu"]
        temp = obs["temp_c"]
        light_obs = obs["lux"]  # Use lux instead of light_umol
        t = obs["t"]
        
        # Maintain temperature buffer for smoothing
        self.temp_buffer.append(temp)
        if len(self.temp_buffer) > 60:
            self.temp_buffer.pop(0)
        temp_smooth = np.mean(self.temp_buffer)
        
        # Maintain OD buffer for smoothing
        self.od_buffer.append(card["od_est"])
        if len(self.od_buffer) > 60:
            self.od_buffer.pop(0)
        od_smooth = np.mean(self.od_buffer)
        
        # Maintain light buffer for smoothing
        self.light_buffer.append(light_obs)
        if len(self.light_buffer) > 60:
            self.light_buffer.pop(0)
        light_smooth = np.mean(self.light_buffer)
        
        # Maintain harvest fraction buffer for smoothing
        self.harvest_buffer.append(card["harvest_frac"])
        if len(self.harvest_buffer) > 60:
            self.harvest_buffer.pop(0)
        harvest_smooth = np.mean(self.harvest_buffer)
        
        # Calculate temperature adjustment
        temp_adjust = self.p["temp_gain"] * (self.p["temp_cap"] - temp_smooth)
        
        # Calculate OD adjustment
        od_adjust = self.p["od_gain"] * (self.p["od_cap"] - od_smooth)
        
        # Adjust light based on temperature and OD
        light_control = light_smooth + temp_adjust + od_adjust
        
        # Clamp light to 0-2000
        light_control = np.clip(light_control, 0.0, 2000.0)
        
        # Smooth light control over time
        light_control = 0.8 * self.last_light + 0.2 * light_control
        self.last_light = light_control
        
        # Calculate harvest fraction based on OD
        od_deviation = od_smooth / self.p["setpoint"] - 1.0
        frac = self.p["gain"] * od_deviation
        frac = np.clip(frac, 0.0, self.p["cap"])
        
        # Adjust harvest fraction based on temperature and OD
        frac += (self.p["temp_gain"] * (self.p["temp_cap"] - temp_smooth))
        frac += (self.p["od_gain"] * (self.p["od_cap"] - od_smooth))
        frac = np.clip(frac, 0.0, self.p["cap"])
        
        # Smooth harvest fraction over time
        frac = 0.8 * self.last_frac + 0.2 * frac
        self.last_frac = frac
        
        # Adjust stirring based on temperature and OD
        stir_control = self.p["stir"]
        if temp_smooth > self.p["temp_cap"] - 1:
            stir_control = max(stir_control - 10, 50)
        elif temp_smooth < self.p["temp_cap"] - 2:
            stir_control = min(stir_control + 10, 200)
        
        # Ensure all actions are within bounds
        light_control = np.clip(light_control, 0.0, 2000.0)
        stir_control = np.clip(stir_control, 50.0, 200.0)
        frac = np.clip(frac, 0.0, 0.5)
        
        return stir_control, light_control, frac
