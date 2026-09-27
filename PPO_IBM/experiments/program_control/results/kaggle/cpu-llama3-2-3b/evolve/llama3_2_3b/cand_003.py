class Controller:
    def __init__(self, params=None):
        self.p = dict(
            defaults={
                "stir": 65.0,  # default stir (50-200)
                "light": 1400.0,  # default light (0-2000)
                "gain": 1.0,  # default gain (0.1-10.0)
                "cap": 0.3,  # default harvest cap (0.0-0.5)
                "turb_per_od": 250.0,  # default turbidity per OD (50-500)
                "temp_setpoint": 35.0,  # default temperature setpoint (25-40)
                "lux_setpoint": 30.0,  # default light setpoint (0-30)
            },
            **(params or {})
        )
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        # Apply gains with a non-linear factor to improve performance
        stir = np.clip(p["defaults"]["stir"] * np.tanh((card["temp_c"] - p["defaults"]["temp_setpoint"]) / 5.0), 50, 200)
        light = np.clip(p["defaults"]["light"] * np.tanh((card["lux"] - p["defaults"]["lux_setpoint"]) / 10.0), 0, 2000)
        frac = float(np.clip(p["defaults"]["gain"] * (card["od_est"] / p["defaults"]["setpoint"] - 1.0), 0.0, p["defaults"]["cap"]))

        # Add a penalty for too high light intensity
        light_penalty = -10.0 * np.abs(card["lux"] - p["defaults"]["lux_setpoint"])
        light += light_penalty

        return stir, light, frac
