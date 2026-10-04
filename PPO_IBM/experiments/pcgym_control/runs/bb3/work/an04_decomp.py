"""Where does the cost come from? Split measured squared error into start-up, setpoint-change windows, rest."""
import numpy as np, sys
from an_common import load, results
R = {r['batch']: r for r in results()}
W = 10
tot = np.zeros(4); nb = 0
for k, d in load(sys.argv[1]).items():
    e2 = ((d['Ca'] - d['Ca_sp']) / 0.01) ** 2 - 0.04   # remove analyser noise floor (sd 0.002)
    ch = np.nonzero(np.diff(d['Ca_sp']))[0] + 1
    m = np.zeros(120, int); m[:W] = 1
    for c in ch: m[c:c + W] = 2
    parts = [e2[m == i].sum() / 120 for i in (1, 2, 0)]
    du = np.std(np.diff(d['Tc']))
    print(k, 'cost %.3f | start %.3f spchg %.3f rest %.3f | sd(dTc) %.2f | sp steps' % (R[k]['cost'], *parts, du),
          np.round(np.diff(d['Ca_sp'])[ch - 1], 3), 'Ca0-sp %.3f' % (d['Ca'][0] - d['Ca_sp'][0]))
    if len(sys.argv) > 2:
        big = np.nonzero((e2 > 1.0) & (m == 0))[0]; print('   large rest errors at', big)
    tot += np.array([R[k]['cost']] + parts); nb += 1
print('MEAN cost %.3f | start %.3f spchg %.3f rest %.3f' % tuple(tot / nb), 'n', nb)
