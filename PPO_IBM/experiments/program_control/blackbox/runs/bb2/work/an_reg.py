# Regression of block log-turbidity slopes on actuator level with batch fixed effects + linear time trend.
import sys
from analyze import *
pat = sys.argv[1]; key = sys.argv[2]; skip = int(sys.argv[3]) if len(sys.argv) > 3 else 1
B = []; rows = []
for r in load():
    if pat not in r['batch']: continue
    c = csvrows(r['batch']); B.append(r['batch'])
    h = c['hour']; y = np.log(np.maximum(c['turbidity_ntu'], 1)); a = c[key]
    start = 0
    for i in range(1, len(h) + 1):
        if i == len(h) or a[i] != a[start]:
            idx = np.arange(start + skip, i)
            if len(idx) >= 2:
                s = np.polyfit(h[idx], y[idx], 1)[0]
                rows.append((len(B) - 1, a[start], h[start], np.log(c['turbidity_ntu'][start]), s, c['temp_c'][idx].mean()))
            start = i
R = np.array(rows); levs = np.unique(R[:, 1])
X = []
for b, l, hh, lt, s, t in R:
    X.append([1.0 * (b == j) for j in range(len(B))] + [hh / 100] + [1.0 * (l == L) for L in levs[1:]])
X = np.array(X); y = R[:, 4]
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ coef; s2 = res @ res / (len(y) - X.shape[1]); cov = s2 * np.linalg.inv(X.T @ X)
print('time trend per 100h: %.3f%%/h' % (coef[len(B)] * 100))
print(f'{key}={levs[0]:.0f}: reference')
for k, L in enumerate(levs[1:]):
    j = len(B) + 1 + k
    print(f'{key}={L:.0f}: {coef[j]*100:+.2f}%/h  se {np.sqrt(cov[j,j])*100:.2f}')
print('batch means %/h:', ' '.join('%.2f' % (coef[j] * 100) for j in range(len(B))))
