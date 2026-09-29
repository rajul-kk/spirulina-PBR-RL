class Controller:
    def __init__(self, params=None):
        self.p = dict(
            Defaults={"stir": 150.0, "light": 1800.0, "setpoint": 0.7, "gain": 2.0, "cap": 0.4, "turb_per_od": 250.0},
            **(params or {})
        )
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_steps = 600
        self.harvest_frac_sum = 0
        self.harvest_frac_count = 0
        self.harvest_times = []
        self.last_harvest_time = 0

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
            if card["t"] - self.last_harvest_time >= 7200 and card["t"] >= 144:
                print(f"Batch complete at time {card['t']} seconds")
                self.last_harvest_time = card["t"]
                return p["stir"], p["light"], harvest_frac
            else:
                return p["stir"], p["light"], frac
        return p["stir"], p["light"], frac

