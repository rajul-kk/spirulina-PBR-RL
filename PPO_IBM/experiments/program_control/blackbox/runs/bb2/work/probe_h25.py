# Open-loop probe: constant stir/light, no harvest. Values set below.
STIR = 100.0
LIGHT = 1200.0
HARV = 0.25
class Controller:
    def __init__(self, params=None):
        pass
    def act(self, obs):
        return (STIR, LIGHT, HARV)
