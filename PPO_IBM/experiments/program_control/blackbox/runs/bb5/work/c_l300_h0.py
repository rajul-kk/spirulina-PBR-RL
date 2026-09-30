class Controller:
    STIR = 100.0
    LIGHT = 300.0
    HARV = 0.0
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (self.STIR, self.LIGHT, self.HARV)
