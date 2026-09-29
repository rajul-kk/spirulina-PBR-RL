import evalm, sys
from evalm import evaluate
from opt3 import P3
evalm.DP = 0.08
class P5(P3):
    def __init__(self, chi=2.0, clo=1.2, brpm=200, **kw):
        super().__init__(**kw); self.chi, self.clo, self.b, self.on = chi, clo, brpm, False
    def __call__(self, t, X, C, T):
        rpm, L, f = super().__call__(t, X, C, T)
        if C > self.chi: self.on = True
        if C < self.clo: self.on = False
        return (self.b if self.on else rpm), L, f
cfgs = {"ref b24x2": lambda: P3(od_sp=1.7, k_final=3, burst_P=24, burst_B=2)}
for chi, clo in [(1.5, 1.1), (2.0, 1.2), (2.5, 1.3), (3.0, 1.5), (1.8, 1.4)]:
    cfgs[f"C{chi}/{clo}"] = (lambda chi=chi, clo=clo: P5(chi=chi, clo=clo, od_sp=1.7, k_final=3))
base = None
for name, mk in cfgs.items():
    r = evaluate(mk, n0s=(50, 200, 1500, 5000), ns=8)
    if base is None: base = r
    print(f"{name:14s}", " ".join(f"n0={k}: {v.mean():7.0f} ({(v/base[k]).mean()*100-100:+.1f}%)" for k, v in r.items()), flush=True)
