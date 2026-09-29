STIR = 200.0
LIGHT = 1800.0
HARV = 0.10
class Controller:
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (STIR, LIGHT, HARV)
