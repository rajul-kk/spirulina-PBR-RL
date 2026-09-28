# per-12h-interval growth: P = (DW_{k+1} - DW_k*(1-f_k))/12 ; f_k = mean requested frac in interval before harvest k
import json, sys, pandas as pd, numpy as np
R = [json.loads(l) for l in open('../trials/results.jsonl')]
pat = sys.argv[1]
out = []
for r in R:
    if pat not in r['batch']: continue
    df = pd.read_csv(f"../trials/{r['batch']}.csv")
    H = r['harvests']
    for k in range(len(H) - 1):
        t0, t1 = H[k]['hour'], H[k+1]['hour']
        f = df.harvest_frac[(df.hour >= t0 - 12) & (df.hour < t0)].mean()
        seg = df[(df.hour >= t0) & (df.hour < t1)]
        x0 = H[k]['lab_dry_weight_mg_per_L'] * (1 - f); x1 = H[k+1]['lab_dry_weight_mg_per_L']
        out.append(dict(b=r['batch'][:4], t=t0, light=seg.light_umol.mean(), stir=seg.stir_rpm.mean(),
                        T=seg.temp_c.mean(), x0=x0, x1=x1, P=(x1 - x0) / 12, mu=np.log(x1 / x0) / 12))
O = pd.DataFrame(out)
print(O.round(3).to_string())
print(O.groupby('light')[['P', 'mu', 'T']].agg(['mean', 'std', 'count']).round(3))
