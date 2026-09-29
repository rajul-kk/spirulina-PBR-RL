STIR = 120.0
LIGHT = 1800.0
HARV = 0.20
class Controller:
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (STIR, LIGHT, HARV)
