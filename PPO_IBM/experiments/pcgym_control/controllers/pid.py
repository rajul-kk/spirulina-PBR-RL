"""Baseline: velocity-form PI(D) on the Ca error, acting on the jacket temperature.
A higher jacket temperature lowers Ca, so the gain is negative. params: kp, ki, kd, bias."""


class Controller:
    def __init__(self, params=None):
        p = {"kp": 30.0, "ki": 60.0, "kd": 0.0, "bias": 299.0}
        p.update(params or {})
        self.kp, self.ki, self.kd, self.u = p["kp"], p["ki"], p["kd"], p["bias"]
        self.e1 = self.e2 = None
        self.t1 = None

    def act(self, obs):
        e = obs["Ca"] - obs["Ca_sp"]            # positive: too much Ca -> heat the jacket
        if self.e1 is None:
            self.e1 = self.e2 = e
            self.t1 = obs["t_min"]
            return self.u
        dt = max(obs["t_min"] - self.t1, 1e-6)
        self.u += self.kp * (e - self.e1) + self.ki * dt * e + self.kd * (e - 2 * self.e1 + self.e2) / dt
        self.u = min(302.0, max(295.0, self.u))
        self.e2, self.e1, self.t1 = self.e1, e, obs["t_min"]
        return self.u
