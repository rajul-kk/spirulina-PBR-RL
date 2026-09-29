P = dict(stir=50.0, light=1500.0, h_start=96.0, frac=0.5)
class Controller:
    """Probe: constant stir/light, no harvest until h_start hours, then constant frac."""
    def __init__(self, params=None):
        self.p = dict(P)
        if params: self.p.update(params)
    def act(self, obs):
        p = self.p
        hr = obs['t'] * 0.02
        h = p['frac'] if hr >= p['h_start'] else 0.0
        return (p['stir'], p['light'], h)
