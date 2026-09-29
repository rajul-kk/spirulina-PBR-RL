import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None): 
        self.p = dict(
            defaults={
                "stir": 100.0,  # increase stir range
                "light": 1600.0,  # increase light range
                "setpoint": 0.75,  # increase setpoint
                "gain": 2.0,  # increase gain
                "cap": 0.45,  # increase harvest cap
            },
            **(params or {})
        )
        self.ps = ProcessState()
        self.harvest_steps = 600
        self.harvest_frac_sum = 0
        self.harvest_frac_count = 0
        self.harvest_times = []

    def act(self, obs):
        p = self.p
        card = self.ps.update(obs)
        od_ratio = card["od_est"] / 250  # directly use od_per_turb
        frac = float(np.clip(p["gain"] * (od_ratio - p["setpoint"]), 0.0, p["cap"]))
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
        return p["stir"], p["light"], frac
