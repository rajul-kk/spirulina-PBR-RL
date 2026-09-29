# ANCOVA: ln(total harvest) ~ arm + ln(DW at covariate hour) (pre-divergence strain covariate).
import sys
from analyze import *
arms = sys.argv[1].split(','); hcov = float(sys.argv[2]) if len(sys.argv) > 2 else 72
R = []
for r in load():
    for j, a in enumerate(arms):
        if a in r['batch']:
            d = {h['hour']: h['lab_dry_weight_mg_per_L'] for h in r['harvests']}
            R.append((j, np.log(r['total_harvested_mg']), np.log(d[hcov]), r['total_harvested_mg']))
R = np.array(R)
X = np.column_stack([1.0 * (R[:, 0] == j) for j in range(len(arms))] + [R[:, 2] - R[:, 2].mean()])
coef, *_ = np.linalg.lstsq(X, R[:, 1], rcond=None)
res = R[:, 1] - X @ coef; s2 = res @ res / (len(res) - X.shape[1]); cov = s2 * np.linalg.inv(X.T @ X)
print('slope on ln DW(%g h): %.2f, resid sd %.3f' % (hcov, coef[-1], np.sqrt(s2)))
for j, a in enumerate(arms):
    m = R[:, 0] == j
    d = coef[j] - coef[0]; se = np.sqrt(cov[j, j] + cov[0, 0] - 2 * cov[0, j])
    print(f'{a:18s} n={m.sum():2d} raw mean {R[m,3].mean():7.0f} median {np.median(R[m,3]):7.0f}  adj vs {arms[0]}: {100*(np.exp(d)-1):+.1f}% (se {100*se:.1f}%)')
