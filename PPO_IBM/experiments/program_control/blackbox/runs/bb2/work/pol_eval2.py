# Compare target policies under both growth models (old saturating model, new regression model).
import numpy as np
import dp_plan, dp2
def run(X12, g, targets):
    X = X12; tot = 0
    for k in range(1, 12):
        tgt = targets(k); f = 0.0 if X <= tgt else min(0.5, 1 - tgt / X)
        if X * (1 - f) < 40: f = max(0, 1 - 40 / X)
        tot += 20 * f * X; X *= (1 - f)
        if k < 11: X = g(X)
    return tot
def P(hold, k8post=500, first_dump=9):
    return lambda k: hold if k < 8 else (k8post if k == 8 and first_dump == 9 else (0 if k >= first_dump else hold))
pols = {'hold500': P(500), 'hold650': P(650), 'hold850': P(850), 'hold1000': P(1000),
        'hold650 dump@8': P(650, first_dump=8), 'hold650 k8->300': P(650, 300)}
inocs = [30, 100, 200, 400, 1000, 2500, 5000]
for label, gmk, params in [('old', lambda m: (lambda X: dp_plan.grow(X, m)), [0.02, 0.028]),
                           ('new', lambda m: (lambda X: float(dp2.grow(np.array(X), m))), [0.022, 0.028, 0.034])]:
    for m in params:
        g = gmk(m); print(label, m, 'inocs', inocs)
        for n, pol in pols.items():
            print(f'   {n:18s}', ' '.join(f'{run(min(0.75*i, 2400), g, pol):7.0f}' for i in inocs))
