class Controller:
    def __init__(self, params=None):
        self.k = 0
    def act(self, obs):
        k = self.k; self.k += 1
        return [302.0, 300.0, 298.0, 295.0][min(k // 30, 3)]
