import numpy as np, itertools, sys
from sim import simulate, light_terms
XG = np.exp(np.linspace(np.log(5), np.log(4000), 60))
_cache = {}
def lopt_table(rpm, Ks=100., Ki=2500.):
    key = (round(rpm), Ks, Ki)
    if key not in _cache:
        Lg = np.arange(50, 4001, 25.0)
        _cache[key] = np.array([Lg[np.argmax([light_terms(X, L, rpm, 1.0, Ks, Ki)[0] for L in Lg])] for X in XG])
    return _cache[key]
class P2:
    def __init__(self, od_sp=1.2, rpm=75, Lmax=1800, k_final=2, g_pred=0.02, L0=250, ramp_h=3.0):
        self.__dict__.update(dict(od_sp=od_sp, rpm=rpm, Lmax=Lmax, k_final=k_final, g_pred=g_pred, L0=L0, ramp_h=ramp_h))
        self.tab = lopt_table(rpm)
    def reset(self): self.f = 0.0
    def __call__(self, t, X, C, T):
        Lo = np.interp(np.log(X), np.log(XG), self.tab) / C**(-1/3)
        L = min(self.Lmax, Lo)
        if t*0.02 < self.ramp_h:
            L = min(L, self.L0 + (L-self.L0)*t*0.02/self.ramp_h)
        if t % 600 == 1 or t == 0:
            ev = t//600 + 1
            if ev > 11 - self.k_final: self.f = 0.5
            else:
                Xp = X*np.exp(self.g_pred*12)
                self.f = min(max(1 - self.od_sp*300/Xp, 0), 0.5)
        return self.rpm, L, self.f
if __name__ == "__main__":
    if sys.argv[1:] == ["trace"]:
        tot, log = simulate(P2(rpm=70), n0=34, mu_max=0.034, verbose=True)
        for r in log[::2]: print(" ".join(f"{v:8.3f}" for v in r))
        sys.exit()
    for n0 in [40, 200, 1500, 5000]:
        best = []
        for od_sp, kf, rpm in itertools.product([0.75, 1.0, 1.5, 2.0], [1, 2, 3, 4], [60, 75, 90]):
            tot, _ = simulate(P2(od_sp=od_sp, k_final=kf, rpm=rpm), n0=n0, mu_max=0.036)
            best.append((tot, od_sp, kf, rpm))
        best.sort(reverse=True)
        print(n0, [f"{b[0]:.0f} sp{b[1]} kf{b[2]} rpm{b[3]}" for b in best[:6]], "worst", f"{best[-1][0]:.0f}")
