import numpy as np
from process_state import ProcessState

class Controller:
    DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.3,
            "turb_per_od": 250.0}

    def __init__(self, params=None):
        self.p = dict(Controller.DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.t = 0
        self.harvest_frac = np.zeros((7200))
        self.harvest_total = 0
        self.pump_L = 0

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        # Increase stir to improve gas exchange and reduce clumping
        stir = np.clip(p["stir"] + self.t * 0.2, 50, 200)

        # Light intensity is increased to drive growth, but saturates and heats the tank
        # Add a small offset to light to avoid saturation
        light = np.clip(p["light"] + self.t * 2.5, 0, 2000)

        # Harvest fraction is increased over time to maximize biomass
        # Add a small penalty for high harvest fractions to avoid over-harvesting
        frac = np.clip(0.3 + np.sin(self.t / 600) * 0.3, 0.0, p["cap"] - 0.1)

        # Harvest pump volume is increased to remove more biomass
        # Add a small penalty for high harvest fractions to avoid over-dosing nutrients
        if np.mean(self.harvest_frac[-600:]) > 0.4:
            frac -= 0.01 * (np.mean(self.harvest_frac[-600:]) - 0.4)
            self.pump_L = np.clip(self.pump_L + self.t / 600 * 0.1, 0, 150)

        # Ensure harvest pump volume does not exceed 150 L
        self.pump_L = np.clip(self.pump_L, 0, 150)

        if self.t % 600 == 0:
            self.harvest_frac[-600:] = 0.0
            self.t = 0
            self.harvest_total = self.pump_L

        return stir, light, frac, self.pump_L

