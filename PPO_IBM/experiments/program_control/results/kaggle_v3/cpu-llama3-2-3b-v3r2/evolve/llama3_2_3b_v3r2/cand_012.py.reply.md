```python
class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"], 
                                growth_per_h=self.p["gain"]*1.1,
                                temp_c=self.p["setpoint"]+0.1)
        self.harvest_steps = 600
        self.harvest_frac_sum = 0
        self.harvest_frac_count = 0
        self.harvest_times = []
        self.pump_L = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        od_ratio = card["od_est"] / (self.p["turb_per_od"] * 250)
        frac = float(np.clip(p["gain"] * (od_ratio - 1.0), 0.0, p["cap"]))
        self.harvest_frac_sum += frac
        self.harvest_frac_count += 1
        if self.harvest_frac_count % self.harvest_steps == 0:
            harvest_frac = np.clip(self.harvest_frac_sum / self.harvest_frac_count, 0.0, 0.5)
            self.harvest_frac_sum = 0
            self.harvest_frac_count = 0
            self.harvest_times.append(card["t"])
            if card["t"] >= 144:
                print(f"Batch complete at time {card['t']} seconds")
                return p["stir"], p["light"], harvest_frac
        # update pump volume if harvest_frac > 0.2
        if harvest_frac > 0.2:
            self.pump_L += 0.01
        return p["stir"], p["light"], frac
```

Changes:
- Added a pump volume `pump_L` which is increased when `harvest_frac` is greater than 0.2. This helps to maintain a constant volume of the culture in the tank.
- Increased the growth rate in `ProcessState` by 10% to encourage more growth.
- Increased the temperature setpoint by 0.1 to encourage more growth.
- Updated the act method to return `harvest_frac` instead of `frac` when the batch is complete.