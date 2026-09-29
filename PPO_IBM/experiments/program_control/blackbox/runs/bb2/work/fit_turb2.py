# Flexible calibration fit on stir-100 batches: ln NTU = ln(A(1-exp(-X/D))) - c1*t/100 - c2*(t/100)*(X/1000)
from analyze import *
from scipy.optimize import least_squares
P = []
for r in load():
    c = csvrows(r['batch'])
    if abs(c['stir_rpm'].mean() - 100) > 1: continue
    for h in r['harvests']:
        i = int(h['hour']) - 1
        P.append((h['hour'], c['turbidity_ntu'][i], h['lab_dry_weight_mg_per_L']))
t, n, d = np.array(P).T
def model(p, t, d):
    A, D, c1, c2 = p
    return np.log(A * (1 - np.exp(-d / D))) - c1 * t / 100 - c2 * (t / 100) * (d / 1000)
f = least_squares(lambda p: model(p, t, d) - np.log(n), [700, 800, 0.1, 0.1])
res = model(f.x, t, d) - np.log(n)
print('A=%.0f D=%.0f c1=%.3f c2=%.3f rms=%.3f' % (*f.x, np.sqrt(np.mean(res ** 2))))
for lo in range(0, 144, 36):
    for dlo, dhi in [(0, 400), (400, 800), (800, 5000)]:
        m = (t >= lo) & (t < lo + 36) & (d >= dlo) & (d < dhi)
        if m.sum() > 2: print(lo, dlo, dhi, m.sum(), 'resid %.3f' % (-res[m].mean()))
