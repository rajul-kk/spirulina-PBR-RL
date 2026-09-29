class Controller:
    DEFAULTS = {
        "stir": 65.0,
        "light": 1400.0,
        "setpoint": 0.6,
        "gain": 1.0,
        "cap": 0.30,
        "turb_per_od": 250.0
    }

    def __init__(self, params=None):
        self.p = dict(self.DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"], gain=self.p["gain"], cap=self.p["cap"])
        self.harvest_steps = 600
        self.harvest_frac_sum = 0
        self.harvest_frac_count = 0
        self.harvest_times = []

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
                return p["stir"], p["light"], 0.0
        return p["stir"], p["light"], frac
