# Does batch growth rate correlate with broth temperature? mu over 12-60h from lab DW vs mean temp (same window).
from analyze import *
R = []
for r in load():
    d = [h['lab_dry_weight_mg_per_L'] for h in r['harvests']]
    if any(h['harvested_mg'] > 0 for h in r['harvests'][:5]) or len(d) < 5: continue
    c = csvrows(r['batch']); m = (c['hour'] >= 12) & (c['hour'] < 60)
    mu = np.log(d[4] / d[0]) / 48
    R.append((c['light_umol'][m].mean(), c['stir_rpm'][m].mean(), c['temp_c'][m].mean(), d[0], mu, c['temp_c'][:3].mean()))
    print(f"{r['batch']:28s} L={R[-1][0]:5.0f} S={R[-1][1]:4.0f} T={R[-1][2]:.2f} T0={R[-1][5]:.2f} X12={d[0]:6.0f} mu={mu:.4f}")
R = np.array(R)
m = R[:, 3] < 600
X = np.column_stack([np.ones(m.sum()), R[m, 2] - 35, (R[m, 0] - 1200) / 1000, np.log(R[m, 3] / 200)])
coef, *_ = np.linalg.lstsq(X, R[m, 4], rcond=None)
res = R[m, 4] - X @ coef; s2 = res @ res / (m.sum() - 4); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
print('mu = %.4f + %.4f*(T-35) + %.4f*(L-1200)/1000 + %.4f*ln(X12/200); se' % tuple(coef), np.round(se, 4), 'resid sd %.4f' % np.sqrt(s2))
