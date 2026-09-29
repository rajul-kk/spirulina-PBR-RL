class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)

        # Use a more sophisticated light control based on lux and light umol
        light_control = np.clip(p["light_umol"] - card["lux"] / 30, 0, p["light_umol"])
        # Add a saturation term to prevent over-saturation
        light_control = np.clip(light_control, 0, p["light_umol"] * 0.9)

        # Use a more sophisticated stir control based on conductivity and stir rpm
        stir_control = np.clip(p["conductivity"] / 50, 0, p["conductivity_threshold"])
        stir_control = stir_control * p["stir_threshold"] * p["stir_control_coefficient"]
        # Add a term to prevent over-stirring
        stir_control = np.clip(stir_control, 0, p["conductivity_threshold"] * 0.8)

        frac = np.clip(p["gain"] * (card["od_est"] / p["setpoint"] - 1.0), 0.0, p["cap"])
        frac = frac * stir_control

        return p["stir"], light_control, frac

