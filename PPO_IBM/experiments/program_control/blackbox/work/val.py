import json, sys, numpy as np, pandas as pd
R = [json.loads(l) for l in open('../trials/results.jsonl')]
lab = sys.argv[1]
P = pd.DataFrame([dict(inoc=r['inoculum'], tot=r['total_harvested_mg'] / 1000, lost=r['culture_lost']) for r in R if r['batch'].endswith('_' + lab)])
def s(d): return f"n={len(d):3d} median {d.tot.median():6.2f} g  p25 {d.tot.quantile(.25):6.2f}  mean {d.tot.mean():6.2f}  min {d.tot.min():6.2f}  lost {int(d.lost.sum())}"
print('ALL      ', s(P))
for lo, hi in [(0, 100), (100, 200), (200, 400), (400, 1000), (1000, 1e9)]:
    d = P[(P.inoc >= lo) & (P.inoc < hi)]
    if len(d): print(f"{lo:4.0f}-{hi:<6.0f}", s(d))
