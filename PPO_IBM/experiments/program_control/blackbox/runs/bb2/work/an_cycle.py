# Block-slope analysis of within-batch step designs: d ln(turb)/dt per block, by actuator level.
import sys
from analyze import *
pat = sys.argv[1]; key = sys.argv[2]; skip = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rows = []
for r in load():
    if pat not in r['batch']: continue
    c = csvrows(r['batch'])
    h = c['hour']; y = np.log(np.maximum(c['turbidity_ntu'], 1)); a = c[key]
    # find blocks of constant action
    start = 0
    for i in range(1, len(h) + 1):
        if i == len(h) or a[i] != a[start]:
            idx = np.arange(start + skip, i)
            if len(idx) >= 2:
                s = np.polyfit(h[idx], y[idx], 1)[0]
                rows.append((r['batch'], a[start], h[start], c['turbidity_ntu'][start], s, c['temp_c'][idx].mean()))
            start = i
rows = np.array([(x[1], x[2], x[3], x[4], x[5]) for x in rows])
# remove batch-time trend: residual vs mean slope of neighbors
for lev in np.unique(rows[:, 0]):
    m = rows[:, 0] == lev
    for lo, hi in [(0, 300), (300, 600), (600, 2000)]:
        mm = m & (rows[:, 2] >= lo) & (rows[:, 2] < hi)
        if mm.sum():
            print(f'{key}={lev:6.0f} turb[{lo},{hi}) n={mm.sum():3d} slope={rows[mm,3].mean()*100:6.2f}%/h se={rows[mm,3].std()/np.sqrt(mm.sum())*100:5.2f} temp={rows[mm,4].mean():.2f}')
