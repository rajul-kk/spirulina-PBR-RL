import numpy as np
from process_state import ProcessState

class Controller:
    def __init__(self, params=None):
        self.p = dict(turb_per_od=250.0,
                      turb_per_od_multiplier=1.0,
                      stir_per_od=65.0,
                      stir_per_od_multiplier=1.0,
                      light_per_od=1400.0,
                      light_per_od_multiplier=1.0,
                      setpoint=0.6,
                      gain=1.0,
                      cap=0.30,
                      temp_setpoint=35.0,
                      temp_per_od_multiplier=1.0,
                      temp_per_od_offset=0.0,
                      lux_per_od_multiplier=1.0,
                      lux_per_od_offset=0.0)
        self.ps = ProcessState()

    def act(self, obs):
        card = self.ps.update(obs)
        stir = np.clip(self.p["stir_per_od"] * (card["od_est"] / self.p["setpoint"]) - self.p["temp_per_od_offset"] * self.p["temp_per_od_multiplier"],
                      self.p["stir_per_od_multiplier"] * self.p["cap"],
                      self.p["stir_per_od_multiplier"] * 200)
        light = np.clip(self.p["light_per_od"] * (card["lux"] / self.p["lux_per_od_multiplier"]) + self.p["lux_per_od_offset"],
                        0,
                        self.p["light_per_od_multiplier"] * 2000)
        frac = float(np.clip(self.p["gain"] * (card["od_est"] / self.p["setpoint"] - 1.0), 0.0, self.p["cap"]))
        return stir, light, frac

