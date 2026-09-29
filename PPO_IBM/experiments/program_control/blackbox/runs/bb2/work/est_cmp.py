# Replay estimators on hourly logs (stir-100 batches) and compare with lab DW before each harvest.
# E1: v3 estimator (t, t*X clumping). E2: clumping from the running integral of the estimate itself.
from analyze import *
from scipy.optimize import least_squares
import math
def inv(ntu, fn, xmax=3000):
    lo, hi = 0.0, xmax
    # find peak by scan
    xs = np.linspace(1, xmax, 300); v = fn(xs); ip = int(np.argmax(v)); hi = xs[ip]
    if ntu >= v[ip]: return hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if fn(mid) < ntu: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
data = []
for r in load():
    c = csvrows(r['batch'])
    if abs(c['stir_rpm'].mean() - 100) > 1: continue
    data.append((r, c))
def replay(params, kind):
    A, D, a, b = params; errs = []
    for r, c in data:
        I = 0.0; X = None; dws = {int(h['hour']) - 1: h['lab_dry_weight_mg_per_L'] for h in r['harvests']}
        for i, hr in enumerate(c['hour']):
            n = c['turbidity_ntu'][i]
            if kind == 1: fn = lambda x: A * (1 - np.exp(-x / D)) * np.exp(-a * hr / 100 - b * hr / 100 * x / 1000)
            else: fn = lambda x: A * (1 - np.exp(-x / D)) * np.exp(-a * hr / 100 - b * (I + x * 0) / 1e5)
            X = inv(n, fn)
            I += X * 1.0  # 1 h steps, mg/L*h
            if i in dws: errs.append((dws[i], X, hr))
    return np.array(errs)
for kind, p in [(1, [996, 1230, 0.056, 0.182]), (2, [973, 1188, 0.082, 0.196])]:
    E = replay(p, kind); le = np.log(E[:, 1] / E[:, 0])
    print('estimator', kind, 'rms log err %.3f' % np.sqrt(np.mean(le ** 2)))
    for lo, hi in [(0, 200), (200, 500), (500, 800), (800, 1200), (1200, 5000)]:
        m = (E[:, 0] >= lo) & (E[:, 0] < hi)
        print(f'   DW[{lo},{hi}) n={m.sum()} bias {le[m].mean():+.3f} sd {le[m].std():.3f}')
