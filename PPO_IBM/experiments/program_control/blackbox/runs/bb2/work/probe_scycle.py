# Within-batch stir step design: cycle stir in 4 h blocks, light fixed 1600, no harvest.
LEVELS = [100.0, 50.0, 150.0, 200.0]
BLOCK = 200
class Controller:
    def __init__(self, params=None):
        self.k = 0
    def act(self, obs):
        i = (self.k // BLOCK) % len(LEVELS)
        self.k += 1
        return (LEVELS[i], 1600.0, 0.0)
