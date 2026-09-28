import json, pandas as pd, numpy as np
R = [json.loads(l) for l in open('../trials/results.jsonl')]
rows = []
for r in R:
    df = pd.read_csv(f"../trials/{r['batch']}.csv").set_index('hour')
    for h in r['harvests']:
        t = h['hour']
        # mean NTU over the 3 hours before the assay (assay taken just before harvest)
        ntu = df.turbidity_ntu.loc[t-2:t-0.5].mean() if t-0.5 in df.index or True else np.nan
        rows.append(dict(b=r['batch'], t=t, dw=h['lab_dry_weight_mg_per_L'], ntu=ntu,
                         stir=df.stir_rpm.mean(), light=df.light_umol.mean()))
C = pd.DataFrame(rows); C['ratio'] = C.ntu / C.dw
C.to_csv('calib_rows.csv', index=False)
print(C.pivot_table(index='b', columns='t', values='ratio').round(2).to_string())
