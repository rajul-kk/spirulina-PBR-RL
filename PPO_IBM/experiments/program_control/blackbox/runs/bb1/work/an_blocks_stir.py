"""Within-batch block analysis for c07-type experiments: slope of ln(NTU) per 6h block vs light & temp."""
import json, csv, sys
import numpy as np
TR = '../trials/'
sel = sys.argv[1]; blk = float(sys.argv[2]) if len(sys.argv) > 2 else 6
recs = []
batches = [json.loads(l)['batch'] for l in open(TR + 'results.jsonl') if sel in l]
for bi, b in enumerate(batches):
    rows = list(csv.DictReader(open(TR + b + '.csv')))
    hr = np.array([float(x['hour']) for x in rows]); ntu = np.array([float(x['turbidity_ntu']) for x in rows])
    T = np.array([float(x['temp_c']) for x in rows]); L = np.array([float(x['stir_rpm']) for x in rows])
    nb = int(96 // blk)
    for k in range(nb):
        m = (hr >= k*blk + 1) & (hr <= k*blk + blk)
        if m.sum() < 3: continue
        s = np.polyfit(hr[m], np.log(ntu[m]), 1)[0]
        recs.append((bi, L[(hr >= k*blk) & (hr < k*blk+blk)].mean(), T[m].mean(), np.log(ntu[m].mean()), s, k))
R = np.array(recs)
nbat = len(batches)
# design: batch FE, light levels as dummies (ref = lowest), temp, lnNTU
levels = sorted(set(np.round(R[:,1])))
X = [ (R[:,0] == i).astype(float) for i in range(nbat)]
for lv in levels[1:]: X.append((np.round(R[:,1]) == lv).astype(float))
X.append(R[:,2] - R[:,2].mean()); X.append(R[:,3] - R[:,3].mean())
X = np.array(X).T; y = R[:,4]
beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
sig2 = np.sum((y - X@beta)**2) / (len(y) - X.shape[1])
se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T@X)))
names = [f'b{i}' for i in range(nbat)] + [f'L{int(l)}' for l in levels[1:]] + ['temp', 'lnNTU']
for n, bb, s in zip(names, beta, se): print(f'{n:>8} {bb:+.4f} +- {s:.4f}')
print('resid sd', np.sqrt(sig2), 'n', len(y))
for lv in levels:
    m = np.round(R[:,1]) == lv
    print('light', lv, 'mean slope', R[m,4].mean().round(4), 'mean T', R[m,2].mean().round(2), 'n', m.sum())
