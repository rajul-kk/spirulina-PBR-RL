class Controller:
    STIR = 100.0
    LIGHT = 2000.0
    START_H = 12.0
    FRAC = 0.2
    def __init__(self, params=None):
        pass
    def act(self, obs):
        h = obs['t'] * 0.02
        f = self.FRAC if h >= self.START_H else 0.0
        return (self.STIR, self.LIGHT, f)
