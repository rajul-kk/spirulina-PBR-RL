import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = {
            "stir": 100.0,           # Stirring rate (50-200 RPM)
            "light": 1400.0,         # Light intensity (0-2000 µmol)
            "setpoint_od": 0.75,     # Target OD
            "gain": 1.5,             # Harvest gain
            "cap": 0.45,             # Harvest cap
            "turb_per_od": 250.0,    # Turbidity per OD
            "temp_setpoint": 35.0,   # Target temperature
            "temp_gain": 0.8,        # Temperature gain
            "temp_cap": 2.0,         # Temperature cap
            "min_light": 800.0,      # Minimum light intensity
            "max_light": 1600.0,     # Maximum light intensity
            "min_stir": 60.0,        # Minimum stir rate
            "max_stir": 140.0,       # Maximum stir rate
            "harvest_window": 600,   # Number of steps to average harvest
            "harvest_history": [],   # History of harvest fractions
            "growth_threshold": 0.05, # Minimum growth rate to continue
            "harvest_cooldown": 60,   # Number of steps to wait after harvest
            "last_harvest": 0,        # Last harvest step
            "od_est": 0.0,           # Estimated OD
            "od_est_slow": 0.0,      # Estimated OD (slow)
        }
        if params:
            self.p.update(params)
        self.ps = ProcessState(turb_per_od=self.p["turb_per_od"])
        self.harvest_history = []
        self.last_harvest = 0

    def act(self, obs):
        card = self.ps.update(obs)
        od_est = card["od_est"]
        od_est_slow = card["od_est_slow"]
        growth_per_h = card["growth_per_h"]
        temp_c = card["temp_c"]
        hours = card["hours"]
        hours_to_harvest = card["hours_to_harvest"]
        harvests_done = card["harvests_done"]
        pump_L = card["pump_L"]
        ph = card["ph"]
        light_obs_umol = card["light_obs_umol"]
        t = obs["t"]

        # Clamp OD estimates
        od_est = np.clip(od_est, 0, 1.5)
        od_est_slow = np.clip(od_est_slow, 0, 1.5)

        # Harvest fraction
        od_diff = od_est - self.p["setpoint_od"]
        harvest_frac = np.clip(self.p["gain"] * od_diff, 0, self.p["cap"])
        self.harvest_history.append(harvest_frac)
        if len(self.harvest_history) > self.p["harvest_window"]:
            self.harvest_history.pop(0)

        # Average harvest fraction over last 600 steps
        avg_harvest = np.mean(self.harvest_history) if self.harvest_history else 0

        # Temperature control
        temp_diff = temp_c - self.p["temp_setpoint"]
        temp_adjust = self.p["temp_gain"] * temp_diff
        temp_adjust = np.clip(temp_adjust, -self.p["temp_cap"], self.p["temp_cap"])
        temp_adjusted = np.clip(temp_c + temp_adjust, 32.0, 38.0)

        # Light control
        light_adjust = 0.0
        if light_obs_umol < self.p["min_light"]:
            light_adjust = self.p["max_light"] - light_obs_umol
        elif light_obs_umol > self.p["max_light"]:
            light_adjust = light_obs_umol - self.p["max_light"]
        light_adjusted = np.clip(light_obs_umol + light_adjust, 0, 2000)

        # Stirring control
        if growth_per_h < self.p["growth_threshold"]:
            # If growth rate is low, increase stirring to enhance mixing
            stir = np.clip(self.p["min_stir"] + 20 * (od_est - self.p["setpoint_od"]), self.p["min_stir"], self.p["max_stir"])
        else:
            # If growth rate is sufficient, adjust stirring based on harvest history
            if hours_to_harvest > 0 and harvests_done < 2:
                # Increase stirring during growth phase
                stir = np.clip(self.p["min_stir"] + 20 * (od_est - self.p["setpoint_od"]), self.p["min_stir"], self.p["max_stir"])
            else:
                # Decrease stirring during harvest phase
                stir = np.clip(self.p["max_stir"] - 20 * (avg_harvest - 0.3), self.p["min_stir"], self.p["max_stir"])

        # Harvest decision
        if t >= self.last_harvest + self.p["harvest_cooldown"] and avg_harvest > 0.1:
            # Harvest only if cooldown is over and average harvest is above threshold
            harvest_frac = avg_harvest
            self.last_harvest = t
        else:
            harvest_frac = 0.0

        return stir, light_adjusted, harvest_frac
