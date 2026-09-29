# Productivity vs density from lab DW (no-harvest constant-light batches), by light level.
from analyze import *
rows = []
for r in load():
    if any(h['harvested_mg'] > 0 for h in r['harvests']) or 'cycle' in r['batch']: continue
    c = csvrows(r['batch']); L = c['light_umol'][0]
    d = [h['lab_dry_weight_mg_per_L'] for h in r['harvests']]
    # use 24 h spans to reduce assay noise
    for k in range(len(d) - 2):
        rows.append((L, np.sqrt(d[k] * d[k + 2]), np.log(d[k + 2] / d[k]) / 24, (d[k + 2] - d[k]) / 24))
R = np.array(rows)
for L in np.unique(R[:, 0]):
    for lo, hi in [(0, 100), (100, 250), (250, 500), (500, 800), (800, 1200), (1200, 1700), (1700, 3000)]:
        m = (R[:, 0] == L) & (R[:, 1] >= lo) & (R[:, 1] < hi)
        if m.sum() >= 2:
            print(f'L={L:5.0f} X[{lo:4d},{hi:4d}) n={m.sum():3d} mu={R[m,2].mean():.4f}/h  prod={R[m,3].mean():5.1f} mg/L/h')
