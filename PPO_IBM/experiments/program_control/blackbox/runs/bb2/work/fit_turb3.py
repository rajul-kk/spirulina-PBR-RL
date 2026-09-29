# Compare clumping models: (a) c1*t + c2*t*X, (b) c*integral(X dt) from assays, (c) b + c1*t
from analyze import *
from scipy.optimize import least_squares
P = []
for r in load():
    c = csvrows(r['batch'])
    if abs(c['stir_rpm'].mean() - 100) > 1: continue
    H = r['harvests']; I = 0.0; prev_t = 0.0; prev_x = H[0]['lab_dry_weight_mg_per_L'] * 0.9
    for k, h in enumerate(H):
        x = h['lab_dry_weight_mg_per_L']
        I += 0.5 * (prev_x + x) * (h['hour'] - prev_t) / 1e5   # (mg/L)*h /1e5
        i = int(h['hour']) - 1
        P.append((h['hour'], c['turbidity_ntu'][i], x, I, len(P)))
        prev_t = h['hour']; prev_x = x * (1 - min(0.5, h['harvested_mg'] / (20 * x)))
t, n, d, I, _ = np.array(P).T
def base(A, D): return np.log(A * (1 - np.exp(-d / D)))
fits = {
 'a t,tX': (lambda p: base(p[0], p[1]) - p[2] * t / 100 - p[3] * (t / 100) * (d / 1000), [700, 800, .1, .1]),
 'b int':  (lambda p: base(p[0], p[1]) - p[2] * I, [700, 800, .1]),
 'c int+t': (lambda p: base(p[0], p[1]) - p[2] * I - p[3] * t / 100, [700, 800, .1, .05]),
}
for k, (fn, p0) in fits.items():
    f = least_squares(lambda p: fn(p) - np.log(n), p0)
    res = fn(f.x) - np.log(n)
    print(k, np.round(f.x, 4), 'rms %.4f' % np.sqrt(np.mean(res ** 2)))
