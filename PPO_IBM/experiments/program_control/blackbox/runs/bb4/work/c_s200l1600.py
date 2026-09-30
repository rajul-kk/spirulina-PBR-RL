class Controller:
    STIR = 200.0
    LIGHT = 1600.0
    H = 0.0
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (self.STIR, self.LIGHT, self.H)
