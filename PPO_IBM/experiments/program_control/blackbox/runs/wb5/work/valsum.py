import json, os, sys, numpy as np
from ana import T
rs = [json.loads(l) for l in open(os.path.join(T, 'results.jsonl'))]
labels = sys.argv[1:]
rs = [r for r in rs if any(r['batch'].endswith('_' + l) for l in labels)]
bins = [(30, 99, '30-99'), (100, 400, '100-400'), (401, 5000, '401-5000')]
def line(name, sel):
    h = np.array([r['total_harvested_mg'] for r in sel]) / 1000
    lost = sum(r['culture_lost'] or bool(r['error']) for r in sel)
    if len(h): print(f"{name:10s} n={len(h):3d} median {np.median(h):5.1f} g  p25 {np.percentile(h,25):5.1f} g  min {h.min():5.1f}  lost {lost}")
line('all', rs)
for lo, hi, n in bins: line(n, [r for r in rs if lo <= r['inoculum'] <= hi])
