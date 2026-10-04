import math


class Controller:
    """Probe controller: crude P on Ca plus a square-wave dither, to check my model replay."""

    def __init__(self, params=None):
        self.k = 0

    def act(self, obs):
        self.k += 1
        e = obs["Ca"] - obs["Ca_sp"]          # Ca too high -> heat up (more reaction)
        u = 299.5 + 60.0 * e + (1.5 if (self.k // 7) % 2 else -1.5)
        return u
