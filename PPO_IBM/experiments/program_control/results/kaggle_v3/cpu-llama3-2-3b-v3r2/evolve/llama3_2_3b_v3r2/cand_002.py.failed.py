class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.prev_od = None
        self.prev_frac = None
        self.harvest_steps = 600
        self.harvest_frac_sum = 0
        self.harvest_frac_count = 0
        self.harvest_times = []

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        if self.prev_od is None:
            self.prev_od = card["od_est"]
            self.prev_frac = 0
        else:
            od_ratio = card["od_est"] / (self.p["turb_per_od"] * 250)
            frac_diff = od_ratio - self.prev_od
            self.prev_od = od_ratio
            self.prev_frac = np.clip(frac_diff, 0.0, 0.05)  # Adjusted to avoid overshooting
        self.harvest_frac_sum += self.prev_frac
        self.harvest_frac_count += 1
        if self.harvest_frac_count % self.harvest_steps == 0:
            harvest_frac = np.clip(self.harvest_frac_sum / self.harvest_frac_count, 0.0, 0.5)
            self.harvest_frac_sum = 0
            self.harvest_frac_count = 0
            self.harvest_times.append(card["t"])
            if card["t"] >= 144:
                print(f"Batch complete at time {card['t']} seconds")
                return p["stir"], p["light"], harvest_frac
        return p["stir"], p["light"], self.prev_frac
