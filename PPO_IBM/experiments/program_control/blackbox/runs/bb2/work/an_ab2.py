# Interval mu from DW (dilution corrected) regressed on light level with batch FE and ln X.
import sys
from analyze import *
pat = sys.argv[1]; hmax = float(sys.argv[2]) if len(sys.argv) > 2 else 200
rows = []; B = {}
for r in load():
    if pat not in r['batch']: continue
    c = csvrows(r['batch']); H = r['harvests']
    for k in range(len(H) - 1):
        if H[k + 1]['hour'] > hmax: break
        d0, d1 = H[k]['lab_dry_weight_mg_per_L'], H[k + 1]['lab_dry_weight_mg_per_L']
        f = min(0.5, H[k]['harvested_mg'] / (20 * d0)); x0 = d0 * (1 - f)
        m = (c['hour'] >= H[k]['hour']) & (c['hour'] < H[k + 1]['hour'])
        rows.append((B.setdefault(r['batch'], len(B)), np.log(d1 / x0) / 12, np.log(x0 / 200), c['light_umol'][m].mean(), c['temp_c'][m].mean()))
R = np.array(rows); nb = len(B); levs = np.unique(R[:, 3])
X = np.column_stack([[1.0 * (b == j) for j in range(nb)] for b in R[:, 0]] if False else [np.array([[1.0 * (b == j) for j in range(nb)] for b in R[:, 0]]), R[:, 2]] + [1.0 * (R[:, 3] == L) for L in levs[1:]])
coef, *_ = np.linalg.lstsq(X, R[:, 1], rcond=None)
res = R[:, 1] - X @ coef; s2 = res @ res / (len(res) - X.shape[1]); cov = s2 * np.linalg.inv(X.T @ X)
print('n', len(res), 'lnX coef %.4f' % coef[nb])
for j, L in enumerate(levs[1:]):
    print(f'light {L:.0f} vs {levs[0]:.0f}: {coef[nb+1+j]:+.4f} /h se {np.sqrt(cov[nb+1+j,nb+1+j]):.4f}')
for L in levs: print('light', L, 'mean temp %.2f' % R[R[:, 3] == L, 4].mean())
