import numpy as np
from process_state import ProcessState

class Controller:
    DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.3,
               "turb_per_od": 250.0, "turb_offset": 0.0, "light_offset": 0.0, "gain_offset": 0.0}

    def __init__(self, params=None):
        self.p = dict(self.DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"], turb_offset=self.p["turb_offset"],
                                light_offset=self.p["light_offset"], gain_offset=self.p["gain_offset"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        # Increase stir to improve gas exchange and reduce clumping
        stir = np.clip(p["stir"] + obs["t"] * 0.5, 50, 200)

        # Light intensity is increased to drive growth, but saturates and heats the tank
        # Add a small offset to light to avoid saturation
        light = np.clip(p["light"] + obs["t"] * 2, 0, 2000)

        # Harvest fraction is increased over time to maximize biomass
        # Add a small penalty for high harvest fractions to avoid over-harvesting
        frac = np.clip(0.2 + np.sin(obs["t"] / 600) * 0.2, 0.0, p["cap"] - 0.1)

        # Harvest pump volume is increased to remove more biomass
        # Add a small penalty for high harvest fractions to avoid over-dosing nutrients
        if np.mean(obs["harvest_frac"][-600:]) > 0.4:
            frac -= 0.01 * (np.mean(obs["harvest_frac"][-600:]) - 0.4)

        pump_L = obs["pump_L"] + np.clip(obs["t"] / 600 * 0.1, 0, 150)

        # Harvest every 600 steps (12 h) and refill with fresh medium
        if obs["t"] % 600 == 0:
            obs["harvest_frac"][-600:] = 0.0
            obs["t"] = 0
            obs["pump_L"] = pump_L
            obs["cells"][-600:] = 0.0

        return stir, light, frac, pump_L, card["od_est"] / 250.0

