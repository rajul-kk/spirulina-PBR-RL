class Controller:
    def __init__(self, params=None):
        self.Tc = 298.5
    def act(self, obs):
        return self.Tc
