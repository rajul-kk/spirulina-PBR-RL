class Controller:
    DEFAULTS = {
        "stir": 65.0,
        "light": 1400.0,
        "setpoint": 0.6,
        "gain": 1.0,
        "cap": 0.30,
        "turb_per_od": 250.0,
        "temp_per_c": 35.0
    }

    def __init__(self, params=None):
        self.p = dict(self.DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"], temp_per_c=self.p["temp_per_c"])
        self.harvest_steps = 600
        self.harvest_frac_sum = 0
        self.harvest_frac_count = 0
        self.harvest_times = []
        self.last_turb_reading = None
        self.last_light_reading = None

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        od_ratio = card["od_est"] / (self.p["turb_per_od"] * 250)
        frac = float(np.clip(p["gain"] * (od_ratio - 1.0), 0.0, p["cap"]))
        self.harvest_frac_sum += frac
        self.harvest_frac_count += 1
        if self.harvest_frac_count % self.harvest_steps == 0:
            harvest_frac = np.clip(self.harvest_frac_sum / self.harvest_frac_count, 0.0, 0.5)
            self.harvest_frac_sum = 0
            self.harvest_frac_count = 0
            self.harvest_times.append(card["t"])
            if card["t"] >= 144:
                print(f"Batch complete at time {card['t']} seconds")
                return p["stir"], p["light"], harvest_frac
        if self.last_turb_reading is not None and self.last_light_reading is not None:
            light_change = (card["lux"] - self.last_light_reading) / self.last_light_reading
            stir_change = (card["turb"] - self.last_turb_reading) / self.last_turb_reading
            if light_change > 0.1 or stir_change > 0.1:
                self.p["gain"] *= 1.1
            elif light_change < -0.1 or stir_change < -0.1:
                self.p["gain"] *= 0.9
        self.last_turb_reading = card["turb"]
        self.last_light_reading = card["lux"]
        return p["stir"], p["light"], frac

