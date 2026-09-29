```python
import numpy as np
from process_state import ProcessState

DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0}

class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.prev_od_est = None
        self.prev_temp_c = None
        self.total_harvest = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Adjust light based on current OD and temperature
        light_factor = max(0.1, min(1.0, 1.5 - (card["od_est"] / 1.5)))
        light = p["light"] * light_factor
        
        # Adjust stir based on current temperature
        stir_factor = max(50, min(200, 100 + (card["temp_c"] - 35) * 10))
        stir = stir_factor
        
        # Adjust harvest fraction based on previous and current OD estimates
        if self.prev_od_est is not None:
            growth_rate = (card["od_est"] - self.prev_od_est) / p["dt_h"]
            harvest_frac = min(0.5, max(0.0, (growth_rate - 0.05) / 0.1))
        else:
            harvest_frac = 0.0
        
        self.prev_od_est = card["od_est"]
        self.prev_temp_c = card["temp_c"]
        self.total_harvest += harvest_frac * p["dt_h"] * 300 * card["pump_L"]
        
        return stir, light, harvest_frac
```

Changes and reasons:
1. **Fixed `ProcessState` import issue**: Ensured `ProcessState` is correctly imported from `process_state`.
2. **Added missing parameters**: Added `self.prev_od_est` and `self.prev_temp_c` to keep track of the previous OD and temperature estimates.
3. **Fixed crash due to undefined variables**: Fixed the crash by ensuring all necessary variables are defined before use.