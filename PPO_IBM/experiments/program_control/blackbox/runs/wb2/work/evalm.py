"""Evaluate policies in my mean-field model over random strains (common random numbers)."""
import numpy as np, sys
from sim import simulate
from opt3 import P3
def strains(n, seed=0):
    r = np.random.RandomState(seed); out = []
    for _ in range(n):
        out.append(dict(mu_max=max(0.02, r.normal(0.040, 0.006))*0.9, topt=float(np.clip(r.normal(36, 1), 30, 40)),
                        Ks=max(50, r.normal(100, 10)), Ki=max(500, r.normal(2500, 250)), tau=r.uniform(1, 4),
                        T0=r.uniform(32, 38)))
    return out
DP = 0.0
def evaluate(make, n0s=(50, 200, 1500), ns=10, seed=0):
    S = strains(ns, seed); res = {}
    for n0 in n0s:
        res[n0] = np.array([simulate(make(), n0=n0, dens_pen=DP, **s)[0] for s in S])
    return res
if __name__ == "__main__":
    DP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    configs = {
      "base sp2 kf3 L1800": lambda: P3(od_sp=2.0, k_final=3),
      "sp1.5 kf3": lambda: P3(od_sp=1.5, k_final=3),
      "sp1.2 kf3": lambda: P3(od_sp=1.2, k_final=3),
      "burst24x2": lambda: P3(od_sp=2.0, k_final=3, burst_P=24, burst_B=2),
      "burst12x1": lambda: P3(od_sp=2.0, k_final=3, burst_P=12, burst_B=1),
      "sp1.2 kf2": lambda: P3(od_sp=1.2, k_final=2),
      "sp2.5 kf4": lambda: P3(od_sp=2.5, k_final=4),
    }
    base = None
    for name, mk in configs.items():
        r = evaluate(mk)
        if base is None: base = r
        print(f"{name:22s}", " ".join(f"n0={k}: {v.mean():7.0f} ({(v/base[k]).mean()*100-100:+.1f}%)" for k, v in r.items()), flush=True)
