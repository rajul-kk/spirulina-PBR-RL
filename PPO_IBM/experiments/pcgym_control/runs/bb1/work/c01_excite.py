import numpy as np
class Controller:
    """Open-loop identification: random-level, random-hold steps of jacket temperature.
    Safety guard: full cooling if reactor T > 329 K."""
    def __init__(self, params=None):
        self.rng = np.random.default_rng()
        self.left = 0
        self.u = 298.5
    def act(self, obs):
        if self.left <= 0:
            self.left = int(self.rng.integers(3, 26))
            self.u = float(self.rng.choice([295.0, 302.0, self.rng.uniform(295.0, 302.0)]))
        self.left -= 1
        if obs["T"] > 329.0:
            return 295.0
        return self.u
