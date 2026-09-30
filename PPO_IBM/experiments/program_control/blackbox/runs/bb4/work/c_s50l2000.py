class Controller:
    STIR = 50.0
    LIGHT = 2000.0
    H = 0.0
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (self.STIR, self.LIGHT, self.H)
