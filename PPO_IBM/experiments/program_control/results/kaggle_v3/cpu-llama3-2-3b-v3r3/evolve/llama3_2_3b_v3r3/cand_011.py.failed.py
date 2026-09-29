class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.t = 0
        self.harvest_frac = np.zeros((7200))
        self.harvest_total = 0
        self.target_od = 0.75
        self.harvest_threshold = 0.5
        self.harvest_penalty = 0.01

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        # Increase stir to improve gas exchange and reduce clumping
        stir = np.clip(p["stir"] + self.t * 0.5, 50, 200)

        # Light intensity is increased to drive growth, but saturates and heats the tank
        # Add a small offset to light to avoid saturation
        light = np.clip(p["light"] + self.t * 2, 0, 2000)

        # Harvest fraction is increased over time to maximize biomass
        # Add a small penalty for high harvest fractions to avoid over-harvesting
        frac = np.clip(0.2 + np.sin(self.t / 600) * 0.2, 0.0, p["cap"] - 0.1)

        # Harvest pump volume is increased to remove more biomass
        # Add a small penalty for high harvest fractions to avoid over-dosing nutrients
        if np.mean(self.harvest_frac[-600:]) > self.harvest_threshold:
            frac -= self.harvest_penalty * (np.mean(self.harvest_frac[-600:]) - self.harvest_threshold)

        pump_L = self.harvest_total + np.clip(self.t / 600 * 0.1, 0, 150)

        # Harvest every 600 steps (12 h) and refill with fresh medium
        if self.t % 600 == 0:
            self.harvest_frac[-600:] = 0.0
            self.t = 0
            self.harvest_total = pump_L

        # Adjust harvest fraction based on biomass
        biomass = card["cells"]
        if biomass < self.target_od:
            frac -= 0.01 * (self.target_od - biomass)
        elif biomass > self.target_od + 0.5:
            frac -= 0.01 * (biomass - (self.target_od + 0.5))

        return stir, light, frac
