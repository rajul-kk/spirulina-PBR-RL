import math

class Controller:
    P = dict(stir=50.0, S=500.0, g=0.84, a=0.001, target=700.0, L0=700.0, Lk=3.0, Lmax=1800.0,
             Tmax=40.0, prod=8.0, mu=0.02, end_int=(10, 11), end_frac=0.5, steps=600,
             alpha=0.01, alt=None)

    def __init__(self, params=None):
        self.p = dict(self.P)
        if params: self.p.update(params)
        self.ntu = None
        self.k = 0
        self.cur_int = 0
        self.req_sum = 0.0
        self.light = self.p['L0']

    def est(self, ntu, h):
        p = self.p
        S, g = p['S'], p['g'] * max(0.3, 1.0 - p['a'] * h)
        y = min(ntu, 0.93 * S) / S
        return -S / g * math.log(1.0 - y)

    def act(self, obs):
        p = self.p
        k = self.k; self.k += 1
        h = k * 0.02
        N = p['steps']
        idx = k // N
        if idx != self.cur_int:
            # a harvest just happened: dilute our filtered turbidity accordingly
            Fm = self.req_sum / N
            if self.ntu is not None:
                self.ntu *= (1.0 - Fm)
            self.cur_int = idx; self.req_sum = 0.0
        n = float(obs['turbidity_ntu'])
        self.ntu = n if self.ntu is None else self.ntu + p['alpha'] * (n - self.ntu)
        X = self.est(self.ntu, h)
        self.X = X
        # light
        L = min(p['Lmax'], p['L0'] + p['Lk'] * X)
        if p['alt'] is not None and X > 0.6 * p['target']:
            L = p['alt'][idx % 2]
        T = float(obs['temp_c'])
        if T > p['Tmax']:
            L = min(L, max(p['L0'], self.light - 2.0))
        self.light = L
        # harvest
        pos = k - idx * N
        left = N - pos
        rem_h = left * 0.02
        harvest_no = idx + 1
        if harvest_no in p['end_int']:
            F = p['end_frac']
        else:
            Xp = X + min(p['prod'], p['mu'] * X) * rem_h
            F = max(0.0, 1.0 - p['target'] / Xp) if Xp > 0 else 0.0
            F = min(F, 0.5)
        r = (F * N - self.req_sum) / left
        r = min(0.5, max(0.0, r))
        self.req_sum += r
        return (p['stir'], L, r)
