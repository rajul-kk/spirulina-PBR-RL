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
            'turb_per_od': 250.0  # Turbidity to OD conversion factor
        }
        if params:
            self.p.update(params)
        self.ps = ProcessState(turb_per_od=self.p['turb_per_od'])
        self.harvest_history = []  # Store harvest fractions for smoothing
        self.harvest_window = 600  # 600 steps = 12 hours
        self.temp_history = []  # Store temperature for smoothing
        self.temp_window = 600  # 600 steps = 12 hours

    def act(self, obs):
        card = self.ps.update(obs)
        # Smooth harvest fraction over 600 steps
        self.harvest_history.append(card['harvest_frac'])
        if len(self.harvest_history) > self.harvest_window:
            self.harvest_history.pop(0)
        avg_harvest = np.mean(self.harvest_history)
        
        # Smooth temperature over 600 steps
        self.temp_history.append(obs['temp_c'])
        if len(self.temp_history) > self.temp_window:
            self.temp_history.pop(0)
        avg_temp = np.mean(self.temp_history)
        
        # Calculate OD estimate
        od_est = card['od_est']
        
        # Calculate harvest fraction based on OD and temperature
        # Higher gain for more aggressive harvesting, but cap at 0.45
        frac = min(max(self.p['gain'] * (od_est / self.p['setpoint'] - 1.0), 0.0), self.p['cap'])
        
        # Adjust light based on temperature to prevent overheating
        if avg_temp > 37:
            light = max(self.p['light'] - 200, 0)
        elif avg_temp < 35:
            light = min(self.p['light'] + 200, 2000)
        else:
            light = self.p['light']
        
        # Adjust stirring based on temperature
        if avg_temp > 37:
            stir = max(self.p['stir'] - 10, 50)
        elif avg_temp < 35:
            stir = min(self.p['stir'] + 10, 200)
        else:
            stir = self.p['stir']
        
        return stir, light, frac
