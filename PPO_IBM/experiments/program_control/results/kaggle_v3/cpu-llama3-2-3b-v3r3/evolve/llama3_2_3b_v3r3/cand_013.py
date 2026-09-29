class Controller:
    def __init__(self, params=None):
        import numpy as np
        from process_state import ProcessState

        self.p = dict(
            Defaults={
                "stir": 65.0,  # mixing; more stir = better gas exchange but more shear/heat
                "light": 1400.0,  # LED intensity; drives growth but saturates, photo-inhibits and heats the tank
                "setpoint": 0.6,  # OD target, 1 OD unit = 300 mg/L dry weight
                "gain": 1.0,  # scaling factor for harvest fraction
                "cap": 0.3,  # maximum harvest fraction
                "turb_per_od": 250.0  # nephelometer reading ~250 NTU per OD unit at the start of a batch
            },
            **(params or {})
        )
        self.ps = ProcessState(turb_per_od=self.p["Defaults"]["turb_per_od"])
        self.t = 0
        self.harvest_frac = np.zeros((7200))
        self.harvest_total = 0
        self.max_harvest_frac = 0.5
        self.harvest_penalty = 0.01
        self.growth_rate_threshold = 0.1
        self.temp_threshold = 35.5

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        # Increase stir to improve gas exchange and reduce clumping
        stir = np.clip(p["Defaults"]["stir"] + self.t * 0.5, 50, 200)

        # Light intensity is increased to drive growth, but saturates and heats the tank
        # Add a small offset to light to avoid saturation
        light = np.clip(p["Defaults"]["light"] + self.t * 2, 0, 2000)

        # Harvest fraction is increased over time to maximize biomass
        # Add a small penalty for high harvest fractions to avoid over-harvesting
        frac = np.clip(0.2 + np.sin(self.t / 600) * 0.2, 0.0, self.max_harvest_frac - self.harvest_penalty)
        if np.mean(self.harvest_frac[-600:]) > self.max_harvest_frac:
            frac -= self.harvest_penalty * (np.mean(self.harvest_frac[-600:]) - self.max_harvest_frac)

        # Check for growth rate and adjust accordingly
        growth_rate = card["growth_per_h"]
        if growth_rate > self.growth_rate_threshold:
            frac -= 0.01 * (growth_rate - self.growth_rate_threshold)
        if growth_rate < 0:
            frac += 0.01 * (growth_rate + self.growth_rate_threshold)

        # Check for temperature and adjust accordingly
        temp_obs = card["temp_c"]
        if temp_obs < self.temp_threshold:
            frac += 0.01 * (self.temp_threshold - temp_obs)
        if temp_obs > self.temp_threshold + 1:
            frac -= 0.01 * (temp_obs - self.temp_threshold - 1)

        # Harvest pump volume is increased to remove more biomass
        # Add a small penalty for high harvest fractions to avoid over-dosing nutrients
        pump_L = self.harvest_total + np.clip(self.t / 600 * 0.1, 0, 150)
        if np.mean(self.harvest_frac[-600:]) > 0.4:
            frac -= 0.01 * (np.mean(self.harvest_frac[-600:]) - 0.4)
        if pump_L > 100:
            frac -= 0.01 * (pump_L - 100)

        return stir, light, frac
