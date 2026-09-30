class Controller:
    STIR = 50.0
    LIGHT = 1600.0
    H = 0.15
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (self.STIR, self.LIGHT, self.H)
