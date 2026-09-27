```python
import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        # Default parameters: stir_rpm, light_umol, harvest_setpoint, gain, cap, turb_per_od
        self.p = {
            'stir': 100.0,    # Increased stirring for better gas exchange
            'light': 1200.0,  # Lowered light to avoid overheating
            'setpoint': 0.7,  # Target OD (OD 0.75 = 300 mg/L)
            'gain': 1.5,      # Higher gain for more aggressive harvesting
            'cap': 0.45,      # Higher harvest cap to maximize biomass
            'turb_per_od': 250.0,  # Turbidity to OD conversion factor
            'temp_setpoint': 35.0,  # Optimal temperature
            'temp_gain': 0.8,  # Temperature control gain
            'temp_cap': 2.0,   # Temperature control cap
            'od_smoothing_window': 60,  # Smoothing window for OD estimation
            'temp_smoothing_window': 60,  # Smoothing window for temperature
            'harvest_window': 600,  # 600 steps = 12 hours for harvest fraction
        }
        if params:
            self.p.update(params)
        self.ps = ProcessState(turb_per_od=self.p['turb_per_od'])
        self.od_estimates = []  # Store OD estimates for smoothing
        self.temp_estimates = []  # Store temperature estimates for smoothing
        self.harvest_history = []  # Store harvest fractions for averaging
        self.last_harvest = 0.0  # Last harvest fraction to avoid abrupt changes

    def act(self, obs):
        # Smooth OD estimates
        self.od_estimates.append(self.ps.update(obs)['od_est'])
        if len(self.od_estimates) > self.p['od_smoothing_window']:
            self.od_estimates.pop(0)
        od_est = np.mean(self.od_estimates)
        
        # Smooth temperature estimates
        self.temp_estimates.append(obs['temp_c'])
        if len(self.temp_estimates) > self.p['temp_smoothing_window']:
            self.temp_estimates.pop(0)
        avg_temp = np.mean(self.temp_estimates)
        
        # Smooth harvest fraction over 600 steps
        self.harvest_history.append(self.last_harvest)
        if len(self.harvest_history) > self.p['harvest_window']:
            self.harvest_history.pop(0)
        avg_harvest = np.mean(self.harvest_history)
        
        # Calculate harvest fraction based on OD and temperature
        # Higher gain for more aggressive harvesting, but cap at 0.45
        frac = min(max(self.p['gain'] * (od_est / self.p['setpoint'] - 1.0), 0.0), self.p['cap'])
        
        # Adjust harvest fraction to avoid abrupt changes
        frac = np.clip(frac, 0.0, self.p['cap'])
        self.last_harvest = frac
        
        # Adjust light based on temperature to prevent overheating
        if avg_temp > self.p['temp_setpoint'] + 2:
            light = max(self.p['light'] - 200, 0)
        elif avg_temp < self.p['temp_setpoint'] - 2:
            light = min(self.p['light'] + 200, 2000)
        else:
            light = self.p['light']
        
        # Adjust stirring based on temperature
        if avg_temp > self.p['temp_setpoint'] + 2:
            stir = max(self.p['st
```