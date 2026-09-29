# Calibration: lab DW vs turbidity reading at harvest hour (just before harvest), as a function of batch age.
from analyze import *
P = []
for r in load():
    c = csvrows(r['batch'])
    for h in r['harvests']:
        i = int(h['hour']) - 1  # reading 1 h before harvest (before any dilution)
        P.append((h['hour'], c['turbidity_ntu'][i], h['lab_dry_weight_mg_per_L'], r['batch'][:4]))
P = np.array([p[:3] for p in P])
k = P[:, 1] / P[:, 2]
for lo, hi in [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150)]:
    for dlo, dhi in [(0, 300), (300, 800), (800, 1300), (1300, 5000)]:
        m = (P[:, 0] >= lo) & (P[:, 0] < hi) & (P[:, 2] >= dlo) & (P[:, 2] < dhi)
        if m.sum(): print(f'hour [{lo},{hi}) DW[{dlo},{dhi}) n={m.sum():3d} NTU/(mg/L)={k[m].mean():.3f} sd {k[m].std():.3f}')
