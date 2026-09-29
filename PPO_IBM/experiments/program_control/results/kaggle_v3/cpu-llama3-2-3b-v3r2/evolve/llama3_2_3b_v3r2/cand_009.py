class Controller:
    DEFAULTS = {"stir": 65.0, "light": 1400.0, "setpoint": 0.6, "gain": 1.0, "cap": 0.30, 
                "turb_per_od": 250.0, "growth_per_h": 0.5, "target_od": 0.75, 
                "initial_temp": 35.0}

    def __init__(self, params=None):
        self.p = dict(self.DEFAULTS, **(params or {}))
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"], 
                                growth_per_h=self.p["growth_per_h"], 
                                target_od=self.p["target_od"], 
                                initial_temp=self.p["initial_temp"])

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        od_ratio = card["od_est"] / (self.p["turb_per_od"] * 250)
        frac = float(np.clip(p["gain"] * (od_ratio - 1.0), 0.0, p["cap"]))
        self.ps.harvest_frac_sum += frac
        self.ps.harvest_frac_count += 1
        if self.ps.harvest_frac_count % self.p["harvest_steps"] == 0:
            harvest_frac = np.clip(self.ps.harvest_frac_sum / self.ps.harvest_frac_count, 0.0, 0.5)
            self.ps.harvest_frac_sum = 0
            self.ps.harvest_frac_count = 0
            self.ps.harvest_times.append(card["t"])
            if card["t"] >= 144:
                print(f"Batch complete at time {card['t']} seconds")
                return p["stir"], p["light"], 0.0
        return p["stir"], p["light"], frac

class ProcessState:
    def __init__(self, turb_per_od, growth_per_h, target_od, initial_temp):
        self.turb_per_od = turb_per_od
        self.growth_per_h = growth_per_h
        self.target_od = target_od
        self.initial_temp = initial_temp
        self.od_est = 0
        self.od_est_slow = 0
        self.growth_per_hest = 0
        self.temp_c = initial_temp
        self.hours = 0
        self.hours_to_harvest = 0
        self.harvests_done = 0
        self.pump_L = 0
        self.ph = 10
        self.light_obs_umol = 0
        self.harvest_frac_sum = 0
        self.harvest_frac_count = 0
        self.harvest_times = []

    def update(self, obs):
        self.ph = obs["ph"]
        self.light_obs_umol = obs["lux"] / 30
        self.turb = obs["turbidity_ntu"] / 100
        self.temp_c = obs["temp_c"]
        self.hours += 0.02
        self.hours_to_harvest = 72 - self.hours
        if self.hours % 600 == 0:
            self.od_est = self.turb * self.turb_per_od / 250
            self.growth_per_hest = self.growth_per_h
            self.harvest_frac_sum += 1
            self.harvest_frac_count += 1
        od_ratio = self.od_est / (self.turb_per_od * 250)
        return {"od_est": self.od_est, 
                "od_est_slow": self.od_est, 
                "growth_per_h": self.growth_per_hest, 
                "temp_c": self.temp_c, 
                "hours": self.hours, 
                "hours_to_harvest": self.hours_to_harvest, 
                "harvests_done": self.harvest_frac_count, 
                "pump_L": self.pump_L, 
                "ph": self.ph, 
                "light_obs_umol": self.light_obs_umol}

    def reset(self):
        self.od_est = 0
        self.od_est_slow = 0
        self.growth_per_hest = 0
        self.temp_c = self.initial_temp
        self.hours = 0
        self.hours_to_harvest = 72
        self.harvests_done = 0
        self.pump_L = 0
        self.ph = 10
        self.light_obs_umol = 0
        self.harvest_frac_sum = 0
        self.harvest_frac_count = 0
        self.harvest_times = []
