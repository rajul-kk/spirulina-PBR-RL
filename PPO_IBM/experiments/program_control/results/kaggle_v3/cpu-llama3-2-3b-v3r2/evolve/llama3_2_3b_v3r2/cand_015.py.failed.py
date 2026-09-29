class Controller:
    def __init__(self, params=None):
        self.p = dict(
            Defaults=self Defaults(),
            stir=self Defaults().stir,
            light=selfDefaults().light,
            setpoint=selfDefaults().setpoint,
            gain=selfDefaults().gain,
            cap=selfDefaults().cap,
            turb_per_od=selfDefaults().turb_per_od,
            harvest_steps=600,
            harvest_frac_sum=0,
            harvest_frac_count=0,
            harvest_times=[]
        )
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])

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
        if card["t"] % 12 == 0:
            if card["t"] < 144:
                return p["stir"], p["light"], frac
            harvest_frac = np.clip(self.harvest_frac_sum / self.harvest_frac_count, 0.0, 0.5)
            harvest_frac = np.interp(card["t"] / 144, [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132], [0, 0.25, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.5])
            self.harvest_frac_sum = harvest_frac * self.harvest_frac_count
            self.harvest_frac_count = 0
        return p["stir"], p["light"], frac

