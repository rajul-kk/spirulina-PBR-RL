```python
import numpy as np
from collections import deque
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = dict({"stir": 150.0, "light": 1500.0, "setpoint": 0.75, "gain": 0.75, "cap": 0.5, "max_growth_rate": 1.5, "turb_per_od": 250.0}, **(params or {}))
        self.harvest_history = deque(maxlen=600)
        self.last_harvest_time = -600
        self.process_state = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.process_state.update(obs)
        
        # Update harvest history
        if obs["t"] % 600 == 0:
            self.harvest_history.append(obs["pump_L"])
            self.last_harvest_time = obs["t"]
        
        # Calculate the target OD
        target_od = 0.75
        
        # Calculate the OD estimation from turbidity
        od_est = obs["turbidity_ntu"] / 250.0
        
        # Calculate the harvest fraction based on OD and target
        frac = float(np.clip(p["gain"] * (od_est / target_od - 1.0), 0.0, p["cap"]))
        
        # Adjust light intensity based on OD estimation and mean harvest fraction
        if od_est < target_od:
            light_adjustment = 1.0
        else:
            light_adjustment = max(0, 1 - (obs["t"] - self.last_harvest_time) / 600)
        
        light = p["light"] * light_adjustment
        
        # Ensure light stays within bounds
        light = max(0, min(2000, light))
        
        # Adjust stir rate based on temperature
        if obs["temp_c"] > 34:
            stir_adjustment = 1.0
        else:
            stir_adjustment = max(0, (35 - obs["temp_c"]) / 5)
        
        stir = p["stir"] * stir_adjustment
        
        # Ensure stir rate stays within bounds
        stir = max(50, min(200, stir))
        
        # Adjust light intensity based on growth rate
        growth_per_h = card["growth_per_h"]
        growth_adjustment = max(0, (p["max_growth_rate"] - growth_per_h) / p["max_growth_rate"])
        light *= growth_adjustment
        
        # Apply dynamic growth rate adjustment
        growth_rate_adjustment = 1.0 + (growth_per_h - p["max_growth_rate"]) / p["max_growth_rate"]
        stir *= growth_rate_adjustment
        light *= growth_rate_adjustment
        
        return stir, light, frac

```

### Changes and Why:
1. **Increased Stir Rate**: Increased the default stir rate from 100 to 150 RPM to enhance mixing and gas exchange, which is crucial for maintaining high growth rates and preventing biomass aggregation.
2. **Increased Light Intensity**: Increased the default light intensity from 1200 to 1500 umol/m²/s to drive better growth, especially at higher OD values where light is more critical.
3. **Dynamic Growth Rate Adjustment**: Added a dynamic adjustment to both stir rate and light intensity based on the current growth rate compared to the maximum growth rate. This helps the controller adapt to different strains and conditions more effectively.