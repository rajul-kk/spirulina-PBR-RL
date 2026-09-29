# Within-batch light step design: cycle light levels in 4 h blocks, stir fixed, no harvest.
LEVELS = [400.0, 1600.0, 800.0, 2000.0, 1200.0]
BLOCK = 200  # steps = 4 h
STIR = 100.0
class Controller:
    def __init__(self, params=None):
        self.k = 0
    def act(self, obs):
        i = (self.k // BLOCK) % len(LEVELS)
        self.k += 1
        return (STIR, LEVELS[i], 0.0)
