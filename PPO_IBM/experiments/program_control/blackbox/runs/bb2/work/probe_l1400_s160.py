# Open-loop probe: constant stir/light, no harvest. Values set below.
STIR = 160.0
LIGHT = 1400.0
HARV = 0.0
class Controller:
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (STIR, LIGHT, HARV)
