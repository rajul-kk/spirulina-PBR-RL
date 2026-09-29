# Fit NTU = A*(1-exp(-DW/D))*(1-c*hour/144) on pre-harvest points (reading 1 h before harvest).
from analyze import *
P = []
for r in load():
    c = csvrows(r['batch'])
    for h in r['harvests']:
        i = int(h['hour']) - 1
        P.append((h['hour'], c['turbidity_ntu'][i], h['lab_dry_weight_mg_per_L']))
P = np.array(P); t, n, d = P.T
best = None
for A in np.arange(450, 900, 10):
    for D in np.arange(300, 1500, 10):
        for cc in np.arange(0, 0.35, 0.02):
            pred = A * (1 - np.exp(-d / D)) * (1 - cc * t / 144)
            e = np.mean((np.log(n) - np.log(pred)) ** 2)
            if best is None or e < best[0]: best = (e, A, D, cc)
print('rms log err %.3f A=%d D=%d clump=%.2f' % (np.sqrt(best[0]), *best[1:]))
e, A, D, cc = best
pred = A * (1 - np.exp(-d / D)) * (1 - cc * t / 144)
for lo, hi in [(0, 200), (200, 500), (500, 1000), (1000, 1500), (1500, 3000)]:
    m = (d >= lo) & (d < hi)
    if m.sum(): print(lo, hi, m.sum(), 'mean log resid %.3f sd %.3f' % (np.mean(np.log(n[m] / pred[m])), np.std(np.log(n[m] / pred[m]))))
