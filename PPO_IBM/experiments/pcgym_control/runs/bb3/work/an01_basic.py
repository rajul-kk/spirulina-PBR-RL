import numpy as np
from an_common import load, results
R = {r['batch']: r for r in results()}
D = load()
for k, d in D.items():
    c = np.mean(((d['Ca'] - d['Ca_sp']) / 0.01) ** 2)
    sp = d['Ca_sp']; ch = np.nonzero(np.diff(sp))[0] + 1
    print(k, 'logged-cost %.3f reported %.3f' % (c, R[k]['cost']), 'sp', np.round(sp[np.r_[0, ch]], 4), 'at', ch,
          'Ca0 %.3f T0 %.1f' % (d['Ca'][0], d['T'][0]))
# noise estimate from const batches: second differences
for k, d in D.items():
    if 'const' in k:
        print(k, 'sd d2Ca/sqrt6 %.5f' % (np.std(np.diff(d['Ca'], 2)) / 6 ** .5), 'sd d2T/sqrt6 %.3f' % (np.std(np.diff(d['T'], 2)) / 6 ** .5))
