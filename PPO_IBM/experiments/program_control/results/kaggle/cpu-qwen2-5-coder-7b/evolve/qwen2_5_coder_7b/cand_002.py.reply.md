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
        
        return p["stir"], p["light"], frac

```

**Changes and Reasoning:**
1. **Harvest Logic Adjustment:** The controller now considers the average harvest fraction over the last 600 steps. If this average is below 50%, it forces a harvest to avoid the risk of culture extinction. Otherwise, it reduces the harvest fraction to maintain biomass levels.
2. **Initialization of `harvest_history` and `last_harvest`:** Added initialization for `harvest_history` to store the harvest fractions and `last_harvest` to keep track of the last harvest time.
3. **Harvest Decision:** Implemented a more sophisticated harvest decision based on the average harvest fraction over the last 600 steps, enhancing the robustness of the controller.