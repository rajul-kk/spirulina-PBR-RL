import numpy as np
class Controller:
    """Open-loop identification: random jacket steps over the full range, random hold times.
    Safety override: full cooling if T > 331 K."""
    def __init__(self, params=None):
        self.rng = None
        self.hold = 0
        self.u = 298.5
    def act(self, obs):
        if self.rng is None:
            seed = int(abs(obs["Ca"] * 1e9 + obs["T"] * 1e6)) % (2**31)
            self.rng = np.random.RandomState(seed)
        if self.hold <= 0:
            self.u = float(self.rng.uniform(295.0, 302.0))
            self.hold = int(self.rng.randint(4, 22))
        self.hold -= 1
        if obs["T"] > 331.0:
            return 295.0
        return self.u
