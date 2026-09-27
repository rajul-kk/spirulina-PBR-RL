class Controller:
    def __init__(self, params=None):
        self.p = dict(
            stir_range=(50, 200),
            light_range=(0, 2000),
            harvest_frac_range=(0, 0.5),
            turb_per_od=250.0,
            temp_setpoint=35.0,
            gain=1.0,
            cap=0.3,
        )
        self.p.update(params or {})
        self.ps = ProcessState(
            turb_per_od=self.p["turb_per_od"],
            temp_setpoint=self.p["temp_setpoint"],
            gain=self.p["gain"],
            cap=self.p["cap"],
        )

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        frac = float(np.clip(p["gain"] * (card["od_est"] / p["temp_setpoint"] - 1.0), 0.0, p["cap"]))
        stir = self._get_stir(p, card)
        light = self._get_light(p, card)
        return stir, light, frac

    def _get_stir(self, p, card):
        temp_c = card["temp_c"]
        stir = self._linearize(p["stir_range"], p["gain"], temp_c)
        return np.clip(stir, p["stir_range"][0], p["stir_range"][1])

    def _get_light(self, p, card):
        lux = card["lux"]
        light = self._linearize(p["light_range"], p["gain"], lux)
        return np.clip(light, p["light_range"][0], p["light_range"][1])

    def _linearize(self, range, gain, value):
        return range[0] + (range[1] - range[0]) * np.tanh(gain * (value - range[0]))
