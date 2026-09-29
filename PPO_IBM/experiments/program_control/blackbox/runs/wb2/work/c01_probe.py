import math
class Controller:
    """Probe: fixed stir 70, light scaled with turbidity OD estimate, harvest to OD 1.0."""
    def __init__(self, params=None):
        self.od = None; self.f = 0.0
    def act(self, obs):
        t = int(obs["t"])
        od_raw = obs["turbidity_ntu"] / 250.0
        od_raw = od_raw / max(1e-3, 1 - 0.05 * od_raw)
        self.od = od_raw if self.od is None else self.od + 0.05 * (od_raw - self.od)
        X = self.od * 300
        L = min(1800.0, 500.0 + 4.0 * X)
        if t % 600 == 1 or t == 0:
            self.f = min(max(1 - 1.0 / max(self.od, 1e-3), 0.0), 0.5) if t > 0 else 0.0
        return 70.0, L, self.f
