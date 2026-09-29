```python
class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.t = 0
        self.harvested_mg = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip((card["od_est"] - p["setpoint"]) / (card["od_est"] + 1), 0.0, p["cap"]))
        if self.t % 600 == 0:
            self.harvested_mg = self.get_harvested_mg()
            self.t = 0
        stir = np.clip(p["stir"] + self.t * 1, 50, 200)
        light = np.clip(p["light"] + self.t * 10, 0, 2000)
        return stir, light, frac

    def get_harvested_mg(self):
        if self.harvested_mg == 0:
            return np.mean([x["harvested_mg"] for x in self.get_harvest_episodes()])
        return np.mean([x["harvested_mg"] for x in self.get_harvest_episodes()])
    def get_harvest_episodes(self):
        return [x for x in self.get_all_episodes() if x["t_h"] % 600 == 0]
    def get_all_episodes(self):
        return [{"t_h": self.t, "turb": x["turb"], "temp_obs": x["temp_obs"], "ph": x["ph"], "stir": x["stir"], "light": x["light"], "frac": x["frac"], "true_od_over_target": x["true_od_over_target"], "true_temp": x["true_temp"], "cells": x["cells"], "harvested_mg": x["harvested_mg"]} for self.t in range(7200)]
```

Changes:

1. Added the `t` attribute to keep track of the current time step. This allows for the calculation of the harvested biomass every 600 steps.

2. Introduced the `get_harvested_mg` method to calculate the average harvested biomass over the last 600 steps. This is more efficient than calculating the average for each episode.

3. Created the `get_harvest_episodes` method to get the episodes where the harvested biomass should be calculated.

4. Created the `get_all_episodes` method to get all the episodes, including those where the harvested biomass is not calculated.

5. Modified the `act` method to update the stir and light values based on the current time step.

6. Used `np.clip` to ensure that the stir, light, and harvest fraction values are within the allowed ranges.

7. Used `np.mean` to calculate the average harvested biomass for the last 600 steps.