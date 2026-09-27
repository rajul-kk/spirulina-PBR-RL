"""Written process state: a small observer that turns the 6 raw sensor channels into the few
slowly varying quantities a bioreactor controller actually needs to remember.

This is the "process state card" alternative to a recurrent network's hidden state: every field
is named, bounded and inspectable. Controller programs may import it (the harness puts this
folder on sys.path) or keep their own state.

    ps = ProcessState()
    card = ps.update(obs, last_action)   # call once per step, before choosing the action
"""
import math


class ProcessState:
    HARVEST_INTERVAL = 600      # steps between harvest events
    DT_H = 0.02

    def __init__(self, turb_per_od=250.0, fast_span=30, slow_span=600, growth_window_h=2.0):
        self.turb_per_od = turb_per_od
        self.a_fast = 2.0 / (fast_span + 1.0)
        self.a_slow = 2.0 / (slow_span + 1.0)
        self.growth_n = max(2, int(growth_window_h / self.DT_H))
        self.turb_fast = None
        self.turb_slow = None
        self.temp_ema = None
        self._log_hist = []        # log(turb_fast) since the last harvest, for the growth estimate
        self.harvests = 0
        self.last_pump = 0.0
        self.card = {}

    def update(self, obs, last_action=None):
        turb = max(float(obs["turbidity_ntu"]), 1e-3)
        temp = float(obs["temp_c"])
        if self.turb_fast is None:
            self.turb_fast = self.turb_slow = turb
            self.temp_ema = temp
        self.turb_fast += self.a_fast * (turb - self.turb_fast)
        self.turb_slow += self.a_slow * (turb - self.turb_slow)
        self.temp_ema += self.a_fast * (temp - self.temp_ema)

        t = int(obs["t"])
        steps_since_harvest = t % self.HARVEST_INTERVAL
        if t > 0 and steps_since_harvest == 1:      # the event was applied on the previous step
            self.harvests += 1
            self._log_hist = []
        self._log_hist.append(math.log(self.turb_fast))
        if len(self._log_hist) > self.growth_n:
            self._log_hist.pop(0)
        n = len(self._log_hist)
        growth_per_h = ((self._log_hist[-1] - self._log_hist[0]) / ((n - 1) * self.DT_H)) if n > 10 else 0.0

        pump = float(obs["pump_L"])
        self.last_pump = pump
        self.card = {
            "od_est": self.turb_fast / self.turb_per_od,          # biased low as cells clump
            "od_est_slow": self.turb_slow / self.turb_per_od,
            "growth_per_h": growth_per_h,                          # net specific growth, 1/h
            "temp_c": self.temp_ema,
            "hours": t * self.DT_H,
            "hours_to_harvest": (self.HARVEST_INTERVAL - steps_since_harvest) * self.DT_H,
            "harvests_done": self.harvests,
            "pump_L": pump,
            "ph": float(obs["ph"]),
            "light_obs_umol": float(obs["lux"]) / 30.0,
        }
        return self.card
