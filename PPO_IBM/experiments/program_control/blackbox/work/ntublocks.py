# NTU-based growth per 12h interval (between harvests); DW_est = -K ln(1-NTU/1000)
import json, sys, pandas as pd, numpy as np
K = 1230.0
R = [json.loads(l) for l in open('../trials/results.jsonl')]
pat = sys.argv[1]; blk = float(sys.argv[2]) if len(sys.argv) > 2 else 12
out = []
for r in R:
    if pat not in r['batch']: continue
    df = pd.read_csv(f"../trials/{r['batch']}.csv")
    df['x'] = -K * np.log(1 - np.clip(df.turbidity_ntu, 0, 990) / 1000)
    for t0 in np.arange(0, r['hours_run'], blk):
        seg = df[(df.hour >= t0 + 1) & (df.hour <= t0 + blk - 1)]   # skip hour right after harvest
        if len(seg) < 5: continue
        a = np.polyfit(seg.hour, seg.x, 1)
        b = np.polyfit(seg.hour, np.log(np.clip(seg.x, 1, None)), 1)
        out.append(dict(b=r['batch'][:4], t=t0, light=seg.light_umol.mean(), stir=seg.stir_rpm.mean(),
                        T=seg.temp_c.mean(), x=seg.x.mean(), P=a[0], mu=b[0]))
O = pd.DataFrame(out)
if len(sys.argv) > 3: print(O.round(3).to_string())
print(O.groupby(['light', 'stir'])[['P', 'mu', 'T', 'x']].agg(['mean', 'std', 'count']).round(3).to_string())
O.to_csv('ntublocks_last.csv', index=False)
