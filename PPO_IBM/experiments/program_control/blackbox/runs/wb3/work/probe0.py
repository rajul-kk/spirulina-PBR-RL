"""Probe controller v0: constant stir, light ramped with turbidity, conservative harvest."""
import math


class Controller:
    def __init__(self, params=None):
        self.light = 300.0
        self.od = None

    def act(self, obs):
        turb = float(obs.get("turbidity_ntu", 0.0))
        od = turb / 250.0
        self.od = od if self.od is None else self.od + 0.05 * (od - self.od)
        target = min(1500.0, max(300.0, 400.0 + 1200.0 * self.od))
        step = 1.0  # umol per step (50/h)
        if target > self.light:
            self.light = min(target, self.light + step)
        else:
            self.light = max(target, self.light - 5 * step)
        frac = 0.0
        if self.od > 0.9:
            frac = min(0.5, 1.0 - 0.75 / self.od)
        return 80.0, self.light, frac
