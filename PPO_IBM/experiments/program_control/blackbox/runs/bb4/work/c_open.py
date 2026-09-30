class Controller:
    STIR = 120.0
    LIGHT = 400.0
    H = 0.0
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (self.STIR, self.LIGHT, self.H)
