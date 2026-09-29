"""Calibrate turbidity (stir 50) against lab DW: ratio NTU/DW vs hour and density."""
import json, csv, sys
import numpy as np
TR = '../trials/'
P = []
for l in open(TR + 'results.jsonl'):
    r = json.loads(l)
    rows = list(csv.DictReader(open(TR + r['batch'] + '.csv')))
    st = [float(x['stir_rpm']) for x in rows]
    if max(st) > 51: continue
    ntu = {float(x['hour']): float(x['turbidity_ntu']) for x in rows}
    for h in r['harvests']:
        t = h['hour']; n = np.mean([ntu[t - 2], ntu[t - 1]])
        P.append((t, n, h['lab_dry_weight_mg_per_L'], r['batch']))
A = np.array([p[:3] for p in P])
t, n, d = A.T
print('n', len(A))
for lo, hi in [(0, 50), (50, 150), (150, 300), (300, 500), (500, 800), (800, 1500), (1500, 4000)]:
    for tl, th in [(0, 36), (36, 72), (72, 100), (100, 144)]:
        m = (d >= lo) & (d < hi) & (t > tl) & (t <= th)
        if m.sum(): print(f'DW {lo:4d}-{hi:<4d} h {tl:3d}-{th:<3d} n {m.sum():3d} NTU/DW {np.mean(n[m]/d[m]):.3f} sd {np.std(n[m]/d[m]):.3f}')
# fit log-model: ln(NTU) = c0 + c1 ln DW + c2 t + c3 (ln DW)^2
X = np.column_stack([np.ones_like(t), np.log(d), t, np.log(d)**2, t*np.log(d)])
b, *_ = np.linalg.lstsq(X, np.log(n), rcond=None)
print('ln NTU fit coef', b, 'resid sd', np.std(np.log(n) - X@b))
