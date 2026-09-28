class Controller:
    """Baseline: constant actuators. Params: stir, light, h."""
    def __init__(self, params=None):
        p = params or {}
        self.stir = p.get('stir', 120.0)
        self.light = p.get('light', 400.0)
        self.h = p.get('h', 0.2)
    def act(self, obs):
        return (self.stir, self.light, self.h)
