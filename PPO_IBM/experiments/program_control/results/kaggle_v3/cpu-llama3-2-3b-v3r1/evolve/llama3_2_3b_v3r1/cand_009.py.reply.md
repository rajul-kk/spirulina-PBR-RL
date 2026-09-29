```python
import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.p["conductivity_threshold"] = 200
        self.p["conductivity_threshold"] = 200
        self.p["light_control_coefficient"] = 0.7
        self.p["stir_control_coefficient"] = 0.8
        self.p["harvest_control_coefficient"] = 0.9

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Use more sophisticated light control based on lux and light umol
        light_control = np.clip(p["light_control_coefficient"] * (p["light_umol"] - card["lux"] / 30), 0, p["light_umol"])
        
        # Use a more sophisticated stir control based on conductivity and stir rpm
        stir_control = np.clip(p["conductivity_threshold"] * p["stir_control_coefficient"] * p["conductivity"] / 50, 0, p["conductivity_threshold"])
        
        # Use a more sophisticated harvest control based on pump_L and harvest_frac
        harvest_control = np.clip(p["harvest_control_coefficient"] * card["pump_L"] * (1 - card["pump_L"] / 100), 0, p["cap"])
        harvest_frac = np.clip(harvest_control / 100, 0, p["cap"])
        
        # Use a more sophisticated light control based on lux and light umol
        light_control = np.clip(light_control / (p["light_umol"] / 100 + 1), 0, p["light_umol"])
        
        stir_control = stir_control / (p["conductivity"] / 50 + 1)
        stir_control = stir_control * p["stir"]
        
        return np.clip(p["stir"] * stir_control, 50, 200), np.clip(light_control, 0, p["light_umol"]), np.clip(harvest_frac, 0, p["cap"])


DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0, "conductivity_threshold": 200, "light_umol_threshold": 1200,
            "stir_threshold": 150, "stir_control_coefficient": 0.8, "light_control_coefficient": 0.7,
            "harvest_control_coefficient": 0.9}
```

Changes:
1. Adjusted the conductivity threshold to 200 to prevent the stir control from becoming too aggressive.
2. Adjusted the light control coefficient to 0.7 to prevent the light from becoming too intense.
3. Added a harvest control coefficient to 0.9 to increase the harvest fraction.
4. Modified the stir control to prevent it from becoming too aggressive.
5. Modified the light control to prevent it from becoming too intense.
6. Modified the harvest control to prevent it from becoming too high.
7. Modified the stir control to prevent it from becoming too high.
8. Modified the light control to prevent it from becoming too high.
9. Modified the harvest control to prevent it from becoming too high.