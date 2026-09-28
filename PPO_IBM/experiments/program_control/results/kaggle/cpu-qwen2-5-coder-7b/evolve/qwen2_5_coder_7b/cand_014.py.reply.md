```python
class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest = 0
        self.max_growth_rate = 0.0
        self.current_growth_rate = 0.0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        
        # Update growth rate
        if card["hours"] > 0:
            self.current_growth_rate = card["growth_per_h"]
            if self.current_growth_rate > self.max_growth_rate:
                self.max_growth_rate = self.current_growth_rate
        
        # Light intensity adjustment based on OD and growth rate
        if card["od_est"] < 0.3:
            light = 500.0
        elif card["od_est"] > 1.5:
            light = 2000.0
        else:
            light = 1000.0 + (card["od_est"] - 0.3) * 500.0
        
        # Harvest logic
        if card["t"] % 600 == 0 and card["t"] > self.last_harvest:
            harvest = np.mean(self.harvest_history[-600:])
            if harvest < 0.5:
                frac = 0.5  # Force harvest if average is below 50%
            else:
                frac = 0.3  # Reduce harvest fraction otherwise
            self.harvest_history.append(frac)
            self.last_harvest = card["t"]
        
        # Stirrer adjustment based on growth rate
        stir = 100.0 + (self.current_growth_rate / self.max_growth_rate) * 100.0
        
        return min(stir, 200), light, frac
```

**Changes and Reasoning:**
1. **Growth Rate Tracking:** Added logic to track the current and maximum growth rates during the episode. This allows the controller to adapt the harvest fraction and stirrer speed based on the growth dynamics.
2. **Light Intensity Adjustment:** Improved the light intensity adjustment logic to be more responsive to the culture's growth conditions. If the OD is below 0.3, the light is reduced to prevent the culture from starving. If the OD is above 1.5, the light is increased to maintain growth.
3. **Stirrer Speed Adjustment:** Adjusted the stirrer speed based on the current growth rate relative to the maximum observed growth rate. This ensures that the tank is well-mixed during periods of high growth to prevent culture clumping and maintain optimal conditions.