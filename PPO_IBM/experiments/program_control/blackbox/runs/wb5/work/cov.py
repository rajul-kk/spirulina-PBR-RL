import json, math, os, sys, numpy as np
from ana import load, T
res = {json.loads(l)['batch']: json.loads(l) for l in open(os.path.join(T, 'results.jsonl'))}
def mu0(b):
    r = {int(x['hour']): x for x in load(b)}
    return math.log(r[24]['true_od'] / r[6]['true_od']) / 18
def fit(groups):
    X, y, names = [], [], []
    for gi, pat in enumerate(groups):
        for b, r in res.items():
            if b.endswith('_' + pat):
                m = mu0(b); X.append([1, m] + [1 if gj == gi else 0 for gj in range(1, len(groups))]); y.append(r['total_harvested_mg']); names.append((b, round(m, 4), y[-1]))
    X, y = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; s2 = resid @ resid / (len(y) - X.shape[1]); cov = s2 * np.linalg.inv(X.T @ X)
    print('beta', np.round(beta), 'se', np.round(np.sqrt(np.diag(cov))), 'resid sd', round(math.sqrt(s2)))
    return names
if __name__ == '__main__':
    for n in fit(sys.argv[1:]): print(n)
