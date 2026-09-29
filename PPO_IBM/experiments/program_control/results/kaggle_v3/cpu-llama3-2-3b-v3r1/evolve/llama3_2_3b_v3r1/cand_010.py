class Controller:
    def __init__(self, params=None):
        self.p = dict(
            defaults={
                "stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
                "turb_per_od": 250.0, "conductivity_threshold": 150, "light_umol_threshold": 1200,
                "stir_threshold": 150, "stir_control_coefficient": 0.8, "light_control_coefficient": 0.7
            },
            **(params or {})
        )
        self.ps = ProcessState(turb_per_od=self.p["defaults"]["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        # Use a more sophisticated light control based on lux and light umol
        light_control = np.clip(p["defaults"]["light_umol_threshold"] - card["lux"] / 30, 0, p["defaults"]["light_umol_threshold"])
        # Add a saturation term to prevent over-saturation
        light_control = np.clip(light_control, 0, p["defaults"]["light_umol_threshold"] * 0.9)

        # Use a more sophisticated stir control based on conductivity and stir rpm
        stir_control = np.clip(p["defaults"]["conductivity_threshold"] / 50, 0, p["defaults"]["conductivity_threshold"])
        stir_control = stir_control * p["defaults"]["stir_threshold"] * p["defaults"]["stir_control_coefficient"]
        # Add a term to prevent over-stirring
        stir_control = np.clip(stir_control, 0, p["defaults"]["conductivity_threshold"] * 0.8)

        frac = np.clip(p["defaults"]["gain"] * (card["od_est"] / p["defaults"]["setpoint"] - 1.0), 0.0, p["defaults"]["cap"])
        frac = frac * stir_control

        return p["defaults"]["stir"], light_control, frac
