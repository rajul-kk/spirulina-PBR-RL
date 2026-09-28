```python
class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"]))
        
        # Harvest logic
        if card["t"] % 600 == 0 and card["t"] > self.last_harvest:
            harvest = np.mean(self.harvest_history[-600:])
            if harvest < 0.5:
                frac = 0.5  # Force harvest if average is below 50%
            else:
                frac = 0.3  # Reduce harvest fraction otherwise
            self.harvest_history.append(frac)
            self.last_harvest = card["t"]
        
        # Adjust light intensity based on OD
        if card["od_est"] < 0.3:
            p["light"] = 1000.0  # Reduce light intensity to prevent culture from dying out
        else:
            p["light"] = 1400.0  # Increase light intensity for better growth
        
        return p["stir"], p["light"], frac
```

**Changes and Reasoning:**
1. **Harvest Logic Adjustment:** The controller now considers the average harvest fraction over the last 600 steps. If this average is below 50%, it forces a harvest to avoid crash. Otherwise, it reduces the harvest fraction.
2. **Light Intensity Adjustment:** Added logic to adjust the light intensity based on the OD. If the OD is below 0.3, the light intensity is reduced to prevent the culture from dying out. Otherwise, it increases the light intensity for better growth. This ensures the culture stays healthy and grows efficiently.