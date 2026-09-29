import numpy as np
class Controller:
    """Within-batch light experiment: light random from {300,600,1000,1500} in 6 h blocks; stir 50;
    no harvest before 96 h then 0.5."""
    def __init__(self, params=None):
        self.rng = np.random.default_rng()
        self.levels = [300.0, 600.0, 1000.0, 1500.0]
        self.block = -1; self.L = 1400.0
    def act(self, obs):
        hr = obs['t'] * 0.02
        b = int(hr // 6)
        if b != self.block:
            self.block = b; self.L = float(self.rng.choice(self.levels))
        return (50.0, self.L, 0.5 if hr >= 96 else 0.0)
