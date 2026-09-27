```python
import numpy as np

class Controller:
    def __init__(self, params=None):
        self.p = {
            "turb_per_od": 250.0,
            "gain": 1.0,
            "cap": 0.3,
            "temp_setpoint": 35.0,
            "stir_min": 50.0,
            "stir_max": 200.0,
            "light_min": 0.0,
            "light_max": 2000.0,
            "light_setpoint": 30.0,
        }
        self.p.update(params or {})

    def act(self, obs):
        p = self.p
        card = self._update_obs(obs)
        stir = np.clip(p["stir_min"] + (p["stir_max"] - p["stir_min"]) * np.tanh(card["temp_c"] - p["temp_setpoint"]), p["stir_min"], p["stir_max"])
        light = np.clip(p["light_min"] + (p["light_max"] - p["light_min"]) * np.tanh(card["lux"] - p["light_setpoint"]), p["light_min"], p["light_max"])
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["turb_per_od"] - 1.0), 0.0, p["cap"]))
        return stir, light, frac

    def _update_obs(self, obs):
        card = ProcessState(turb_per_od=self.p["turb_per_od"])
        card.update(obs)
        return card
```

Changes and improvements:
- Removed the `DEFAULTS` dictionary from the `__init__` method and instead defined the parameters directly in the `__init__` method. This makes the code more flexible and easier to read.
- Introduced separate parameters for the stir and light control, allowing for more flexibility in the controller's behavior.
- Used the `np.tanh` function to introduce non-linearity in the stir and light controls, which better matches the physical properties of the system.
- Added a `_update_obs` method to encapsulate the process of updating the sensor readings, which makes the code more modular and easier to read.
- Improved the stability of the controller by adding a minimum and maximum value to the stir and light controls, which prevents the stir and light from becoming too extreme.