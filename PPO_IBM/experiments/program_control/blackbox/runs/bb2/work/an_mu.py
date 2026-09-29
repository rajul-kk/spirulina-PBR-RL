# Interval-level growth regression with batch fixed effects:
# mu_k = ln(DW_{k+1} / (DW_k*(1-f_k)))/12 ~ batchFE + X-shape + light + temp terms.
import sys
from analyze import *
rows = []; B = {}
for r in load():
    if r['culture_lost']: continue
    c = csvrows(r['batch']); H = r['harvests']
    for k in range(len(H) - 1):
        d0, d1 = H[k]['lab_dry_weight_mg_per_L'], H[k + 1]['lab_dry_weight_mg_per_L']
        f = min(0.5, H[k]['harvested_mg'] / (20 * d0))
        x0 = d0 * (1 - f)
        m = (c['hour'] >= H[k]['hour']) & (c['hour'] < H[k + 1]['hour'])
        L = c['light_umol'][m].mean(); T = c['temp_c'][m].mean()
        xm = np.sqrt(x0 * d1)
        b = B.setdefault(r['batch'], len(B))
        rows.append((b, np.log(d1 / x0) / 12, xm, L, T))
R = np.array(rows); nb = len(B)
xm = R[:, 2]; L = R[:, 3] / 1000; T = R[:, 4] - 35
feats = {
    'x/1000': xm / 1000, 'L': L, 'L*x/1000': L * xm / 1000, 'L^2': L ** 2,
    'T': T, 'T^2': T ** 2,
}
FE = np.array([[1.0 * (bb == j) for j in range(nb)] for bb in R[:, 0]])
X = np.column_stack([FE] + list(feats.values()))
y = R[:, 1]
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ coef; s2 = res @ res / (len(y) - X.shape[1]); cov = s2 * np.linalg.pinv(X.T @ X)
print('n=%d batches=%d resid sd %.4f' % (len(y), nb, np.sqrt(s2)))
for j, k in enumerate(feats):
    print(f'  {k:10s} {coef[nb+j]:+.4f} se {np.sqrt(cov[nb+j,nb+j]):.4f}')
fe = coef[:nb]; print('batch FE mean %.4f sd %.4f' % (fe.mean(), fe.std()))
np.save('mu_coef.npy', coef[nb:])
