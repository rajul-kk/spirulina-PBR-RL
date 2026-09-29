# Evaluate simple target-type harvest policies on the fitted growth model (relative comparison only).
import numpy as np
from dp_plan import grow
def run(X12, mu0, targets):
    X = X12; tot = 0
    for k in range(1, 12):
        tgt = targets(k)
        f = 0.0 if X <= tgt else min(0.5, 1 - tgt / X)
        if X * (1 - f) < 40: f = max(0, 1 - 40 / X)
        tot += 20 * f * X; X *= (1 - f)
        if k < 11: X = grow(X, mu0)
    return tot
pols = {
 'hold850 k8->500 k9+ dump': lambda k: 850 if k < 8 else (500 if k == 8 else 0),
 'hold850 k9+ dump':          lambda k: 850 if k < 9 else 0,
 'hold850 k8+ dump':          lambda k: 850 if k < 8 else 0,
 'hold700 k8->500 dump':      lambda k: 700 if k < 8 else (500 if k == 8 else 0),
 'hold1000 k8->500 dump':     lambda k: 1000 if k < 8 else (500 if k == 8 else 0),
 'hold850 k7->600 k8->400':   lambda k: 850 if k < 7 else (600 if k == 7 else (400 if k == 8 else 0)),
 'hold850 k10+ dump':         lambda k: 850 if k < 10 else 0,
}
for mu0 in [0.02, 0.028]:
    print('mu0', mu0)
    for name, pol in pols.items():
        vals = [run(min(0.75 * i0, 2400) * np.exp(0), mu0, pol) for i0 in [30, 100, 200, 300, 1000, 5000]]
        print(f'  {name:28s}', ' '.join(f'{v:7.0f}' for v in vals))
