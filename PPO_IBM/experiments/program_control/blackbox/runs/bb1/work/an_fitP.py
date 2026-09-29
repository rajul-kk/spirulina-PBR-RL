"""Fit productivity model dX/dt = a*X/(1+X/K) - m*X to assay intervals (stir 50 batches) and
evaluate harvest policies by DP on the fitted model."""
import json, csv, math, sys
import numpy as np
from scipy.optimize import least_squares
TR = '../trials/'
V = 20.9
data = []
for l in open(TR + 'results.jsonl'):
    r = json.loads(l)
    rows = list(csv.DictReader(open(TR + r['batch'] + '.csv')))
    if np.mean([float(x['stir_rpm']) for x in rows]) > 60: continue
    H = r['harvests']; pump = {float(x['hour']): float(x['pump_L']) for x in rows}
    for k in range(len(H) - 1):
        t0 = H[k]['hour']
        f = max(0, (pump.get(t0 + 1, 0) - pump.get(t0 - 1, 0)) / V)
        x0 = H[k]['lab_dry_weight_mg_per_L'] * (1 - f); x1 = H[k+1]['lab_dry_weight_mg_per_L']
        data.append((x0, x1, r['batch']))
D = np.array([(a, b) for a, b, _ in data])
def sim(x, th, h=12.0, n=60):
    a, K, m = th
    for _ in range(n): x = x + h / n * (a * x / (1 + x / K) - m * x)
    return x
def res(th): return np.log(np.array([sim(x0, th) for x0 in D[:, 0]]) / D[:, 1])
fit = least_squares(res, [0.03, 1000, 0.003], bounds=([0, 10, 0], [1, 1e5, 0.1]))
th = fit.x; print('fit a,K,m', th, 'resid sd', np.std(fit.fun), 'n', len(D))
for X in [30, 100, 200, 400, 800, 1200, 1600, 2000, 3000]:
    print(f'  X {X:5d}  P {th[0]*X/(1+X/th[1]) - th[2]*X:6.2f}  mu {th[0]/(1+X/th[1]) - th[2]:.4f}')
np.save('fit_theta.npy', th)
