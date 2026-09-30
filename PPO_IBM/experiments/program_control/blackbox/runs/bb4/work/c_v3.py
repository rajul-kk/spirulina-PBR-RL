import math

class Controller:
    P = dict(stir=50.0, S=500.0, g=0.84, a=0.001, target=500.0,
             L0=700.0, Lk=5.0, Lmax=2000.0, Tmax=40.5,
             mumax=0.0842, Ks=698.3, Ki=721.9, kx=0.00906, m=0.0008,
             end_int=(9, 10, 11), end_frac=0.5, steps=600, alpha=0.01)

    def __init__(self, params=None):
        self.p = dict(self.P)
        if params: self.p.update(params)
        self.ntu = None
        self.k = 0
        self.cur_int = 0
        self.req_sum = 0.0
        self.light = self.p['L0']
        self.F = 0.0
        self.X = 0.0

    def est(self, ntu, h):
        p = self.p
        S, g = p['S'], p['g'] * max(0.3, 1.0 - p['a'] * h)
        y = min(ntu, 0.93 * S) / S
        return -S / g * math.log(1.0 - y)

    def mu(self, X, L):
        p = self.p
        kx = max(p['kx'] * X, 1e-9)
        f = 0.0
        for i in range(6):
            I = L * math.exp(-kx * (i + 0.5) / 6.0)
            f += I / (p['Ks'] + I + I * I / p['Ki'])
        return p['mumax'] * f / 6.0 - p['m']

    def Lfun(self, X):
        p = self.p
        return min(p['Lmax'], p['L0'] + p['Lk'] * X)

    def predict(self, X, hours):
        n = max(1, int(hours / 0.5))
        dt = hours / n
        for _ in range(n):
            X *= math.exp(self.mu(X, self.Lfun(X)) * dt)
        return X

    def act(self, obs):
        p = self.p
        k = self.k; self.k += 1
        h = k * 0.02
        N = p['steps']
        idx = k // N
        if idx != self.cur_int:
            Fm = self.req_sum / N
            if self.ntu is not None:
                self.ntu *= (1.0 - Fm)
            self.cur_int = idx; self.req_sum = 0.0
        n = float(obs['turbidity_ntu'])
        self.ntu = n if self.ntu is None else self.ntu + p['alpha'] * (n - self.ntu)
        X = self.est(self.ntu, h)
        self.X = X
        L = self.Lfun(X)
        T = float(obs['temp_c'])
        if T > p['Tmax']:
            L = min(L, max(p['L0'], self.light - 2.0))
        elif L > self.light + 20.0:
            L = self.light + 20.0
        self.light = L
        pos = k - idx * N
        left = N - pos
        harvest_no = idx + 1
        if harvest_no > 11:
            F = 0.0
        elif harvest_no in p['end_int']:
            F = p['end_frac']
        else:
            if pos % 10 == 0:
                Xp = self.predict(X, left * 0.02)
                self.F = min(0.5, max(0.0, 1.0 - p['target'] / Xp))
            F = self.F
        r = (F * N - self.req_sum) / left
        r = min(0.5, max(0.0, r))
        self.req_sum += r
        return (p['stir'], L, r)
