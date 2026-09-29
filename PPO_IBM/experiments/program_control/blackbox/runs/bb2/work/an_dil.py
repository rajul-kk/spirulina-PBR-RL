# Is growth faster right after a harvest/refill? interval mu ~ batchFE + lnX + L + f_k (fraction just removed)
from analyze import *
rows = []; B = {}
for r in load():
    if r['culture_lost']: continue
    c = csvrows(r['batch']); H = r['harvests']
    for k in range(len(H) - 1):
        d0, d1 = H[k]['lab_dry_weight_mg_per_L'], H[k + 1]['lab_dry_weight_mg_per_L']
        f = min(0.5, H[k]['harvested_mg'] / (20 * d0)); x0 = d0 * (1 - f)
        m = (c['hour'] >= H[k]['hour']) & (c['hour'] < H[k + 1]['hour'])
        rows.append((B.setdefault(r['batch'], len(B)), np.log(d1 / x0) / 12, x0 / 1000, c['light_umol'][m].mean() / 1000, f, H[k]['hour'] / 100))
R = np.array(rows); nb = len(B)
FE = np.array([[1.0 * (b == j) for j in range(nb)] for b in R[:, 0]])
names = ['x', 'x^2', 'L', 'L*x', 'f', 'hour']
X = np.column_stack([FE, R[:, 2], R[:, 2] ** 2, R[:, 3], R[:, 3] * R[:, 2], R[:, 4], R[:, 5]])
coef, *_ = np.linalg.lstsq(X, R[:, 1], rcond=None)
res = R[:, 1] - X @ coef; s2 = res @ res / (len(res) - X.shape[1]); cov = s2 * np.linalg.pinv(X.T @ X)
print('n', len(res), 'resid sd %.4f' % np.sqrt(s2))
for j, nm in enumerate(names): print(f'  {nm:5s} {coef[nb+j]:+.4f} se {np.sqrt(cov[nb+j,nb+j]):.4f}')
