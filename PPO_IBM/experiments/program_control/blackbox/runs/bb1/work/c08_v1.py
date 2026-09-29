P = dict(stir=50.0, L_lo=800.0, L_hi=1500.0, ntu_lo=150.0, ntu_hi=280.0, T_cap=37.0,
         h_start=96.0, frac=0.5)
class Controller:
    """v1: stir 50; light scheduled on filtered turbidity (800 -> 1500), capped by a temperature
    servo; hold biomass then drain 0.5 from h_start."""
    def __init__(self, params=None):
        self.p = dict(P)
        if params: self.p.update(params)
        self.ntu = None; self.T = None; self.Lcap = 2000.0
    def act(self, obs):
        p = self.p
        hr = obs['t'] * 0.02
        n = obs['turbidity_ntu']; T = obs['temp_c']
        self.ntu = n if self.ntu is None else self.ntu + 0.02 * (n - self.ntu)
        self.T = T if self.T is None else self.T + 0.02 * (T - self.T)
        # temperature servo on the light ceiling (integral action, ~umol per step)
        self.Lcap = min(2000.0, max(300.0, self.Lcap + 2.0 * (p['T_cap'] - self.T)))
        a = (self.ntu - p['ntu_lo']) / (p['ntu_hi'] - p['ntu_lo'])
        a = min(1.0, max(0.0, a))
        L = min(p['L_lo'] + a * (p['L_hi'] - p['L_lo']), self.Lcap)
        h = p['frac'] if hr >= p['h_start'] else 0.0
        return (p['stir'], L, h)
