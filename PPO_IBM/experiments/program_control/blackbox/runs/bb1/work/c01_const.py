class Controller:
    """Baseline probe: constant stir/light, harvest 0.2 once OD proxy above target."""
    def __init__(self, params=None):
        self.p = dict(stir=120.0, light=400.0, h_on=190.0, frac=0.2)
        if params: self.p.update(params)
    def act(self, obs):
        p = self.p
        h = p['frac'] if obs['turbidity_ntu'] > p['h_on'] else 0.0
        return (p['stir'], p['light'], h)
