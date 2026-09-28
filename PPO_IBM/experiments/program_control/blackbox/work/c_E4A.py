class Controller:
    B = [(120.0, 1000.0, 0.0), (120.0, 2000.0, 0.0), (120.0, 500.0, 0.0), (120.0, 1500.0, 0.0), (120.0, 1000.0, 0.0), (120.0, 2000.0, 0.0)]
    def __init__(self, params=None): pass
    def act(self, obs):
        k = int(obs['t'] * 0.02 // 24.0)
        return self.B[min(k, len(self.B) - 1)]
