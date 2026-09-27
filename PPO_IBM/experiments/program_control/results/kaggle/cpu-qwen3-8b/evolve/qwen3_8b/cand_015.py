import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        # Default parameters: stir_rpm, light_umol, harvest_setpoint, gain, cap, turb_per_od
        self.p = {
            'stir': 100.0,    # Moderate stirring for gas exchange
            'light': 1200.0,  # Balanced light to avoid overheating
            'setpoint': 0.7,  # Target OD (OD 0.75 = 300 mg/L)
            'gain': 1.2,      # Moderate gain for harvesting
            'cap': 0.4,       # Moderate harvest cap to avoid overharvesting
            'turb_per_od': 250.0,  # Turbidity to OD conversion factor
            'temp_setpoint': 35.0,  # Ideal temperature for growth
            'temp_gain': 0.03,      # Gain for temperature control
            'temp_cap': 1.5,        # Max temperature change per step
            'od_gain': 1.0,         # Gain for OD control
            'od_cap': 0.4,          # Max harvest fraction
            'od_window': 600,       # 600 steps = 12 hours
            'od_history': []
        }
        if params:
            self.p.update(params)
        self.ps = ProcessState(turb_per_od=self.p['turb_per_od'])
        self.od_history = []
        self.temp_history = []

    def act(self, obs):
        card = self.ps.update(obs)
        self.od_history.append(card['od_est'])
        if len(self.od_history) > self.p['od_window']:
            self.od_history.pop(0)
        avg_od = np.mean(self.od_history)
        
        self.temp_history.append(obs['temp_c'])
        if len(self.temp_history) > self.p['od_window']:
            self.temp_history.pop(0)
        avg_temp = np.mean(self.temp_history)
        
        # Temperature control
        temp_diff = avg_temp - self.p['temp_setpoint']
        temp_adjust = self.p['temp_gain'] * temp_diff
        temp_adjust = np.clip(temp_adjust, -self.p['temp_cap'], self.p['temp_cap'])
        light = self.p['light'] + temp_adjust
        light = np.clip(light, 0, 2000)
        
        # OD control
        od_diff = avg_od - self.p['setpoint']
        harvest_frac = self.p['od_gain'] * od_diff
        harvest_frac = np.clip(harvest_frac, 0, self.p['od_cap'])
        
        # Adjust light based on temperature
        if avg_temp > self.p['temp_setpoint'] + 1:
            light = max(light - 200, 0)
        elif avg_temp < self.p['temp_setpoint'] - 1:
            light = min(light + 200, 2000)
        
        # Adjust stirring based on temperature
        if avg_temp > self.p['temp_setpoint'] + 1:
            stir = max(self.p['stir'] - 10, 50)
        elif avg_temp < self.p['temp_setpoint'] - 1:
            stir = min(self.p['stir'] + 10, 200)
        else:
            stir = self.p['stir']
        
        return stir, light, harvest_frac
