import numpy as np
class Controller:
    """Within-batch stir experiment: stir random from {50,100,200} in 6 h blocks; light 1400;
    no harvest before 96 h then 0.5.  (light column kept constant; stir logged)"""
    def __init__(self, params=None):
        self.rng = np.random.default_rng(); self.block = -1; self.S = 50.0
    def act(self, obs):
        hr = obs['t'] * 0.02
        b = int(hr // 6)
        if b != self.block:
            self.block = b; self.S = float(self.rng.choice([50.0, 100.0, 200.0]))
        return (self.S, 1400.0, 0.5 if hr >= 96 else 0.0)
