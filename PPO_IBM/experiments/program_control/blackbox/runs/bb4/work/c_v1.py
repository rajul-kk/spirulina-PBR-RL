import math

class Controller:
    P = dict(stir=50.0, S=500.0, g=0.82, target=450.0, L0=700.0, Lk=3.0, Lmax=1800.0,
             Tmax=39.5, prod=8.0, end_int=(10, 11), end_frac=0.5, steps=600)

    def __init__(self, params=None):
        self.p = dict(self.P)
        if params: self.p.update(params)
        self.ntu = None
        self.k = 0
        self.cur_int = 0
        self.req_sum = 0.0
        self.req_n = 0
        self.light = self.p['L0']

    def est(self, ntu):
        S, g = self.p['S'], self.p['g']
        y = min(ntu, 0.95 * S) / S
        return -S / g * math.log(1.0 - y)

    def act(self, obs):
        p = self.p
        k = self.k; self.k += 1
        n = float(obs['turbidity_ntu'])
        self.ntu = n if self.ntu is None else self.ntu + 0.05 * (n - self.ntu)
        X = self.est(self.ntu)
        # light
        L = min(p['Lmax'], p['L0'] + p['Lk'] * X)
        T = float(obs['temp_c'])
        if T > p['Tmax']:
            L = max(p['L0'], self.light - 5.0)
        self.light = L
        # harvest
        N = p['steps']
        idx = k // N
        if idx != self.cur_int:
            self.cur_int = idx; self.req_sum = 0.0; self.req_n = 0
        pos = k - idx * N
        rem_h = (N - pos) * 0.02
        harvest_no = idx + 1  # harvest at end of this interval (1..11)
        if harvest_no in p['end_int']:
            F = p['end_frac']
        else:
            Xp = X + p['prod'] * rem_h
            F = max(0.0, 1.0 - p['target'] / Xp) if Xp > 0 else 0.0
            F = min(F, 0.5)
        left = N - pos
        r = (F * N - self.req_sum) / left
        r = min(0.5, max(0.0, r))
        self.req_sum += r
        return (p['stir'], L, r)
