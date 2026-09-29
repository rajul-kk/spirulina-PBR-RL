class Controller:
    def __init__(self, params=None): 
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        # Improved light control based on lux and light umol
        light_control = min(p["light_umol"], 1000 - card["lux"] / 30)
        light_control = np.clip(light_control, 0, p["light_umol"])
        light_control = np.exp(-light_control / (p["light_umol"] - 500))  # Sigmoidal response

        # Improved stir control based on conductivity and stir rpm
        stir_control = min(p["conductivity_threshold"], p["stir_threshold"] * p["conductivity"] / 50)
        stir_control = np.clip(stir_control, 0, p["conductivity_threshold"])
        stir_control = stir_control / (p["conductivity"] / 50 + 1)
        stir_control = stir_control * p["stir"]

        # Improved harvest fraction control
        frac = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"])
        frac = frac * stir_control

        # Harvest signal with decay
        harvest_frac = np.cumprod([frac], 0)
        harvest_frac = np.clip(harvest_frac, 0, 0.5)

        # Harvest signal at 600 steps
        if p["t"] % 600 == 0:
            harvest_frac = np.mean(harvest_frac[-600:])

        return p["stir"], light_control, harvest_frac


DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30,
            "turb_per_od": 250.0, "conductivity_threshold": 150, "light_umol_threshold": 1200,
            "stir_threshold": 150, "stir_control_coefficient": 0.8, "light_control_coefficient": 0.7,
            "light_decay_coefficient": 0.01, "harvest_decay_coefficient": 0.01}
