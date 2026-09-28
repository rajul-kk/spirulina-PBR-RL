# whole-batch: linear slope of lab DW vs time (mg/L/h), mean temp, for h0 batches
import json, sys, pandas as pd, numpy as np
R = [json.loads(l) for l in open('../trials/results.jsonl')]
pat = sys.argv[1]
for r in R:
    if pat not in r['batch']: continue
    df = pd.read_csv(f"../trials/{r['batch']}.csv")
    t = np.array([h['hour'] for h in r['harvests']]); x = np.array([h['lab_dry_weight_mg_per_L'] for h in r['harvests']])
    a = np.polyfit(t, x, 1); m = np.polyfit(t[:4], np.log(x[:4]), 1)
    print(f"{r['batch']:24s} DW12 {x[0]:5.0f} DW132 {x[-1]:5.0f} slope {a[0]:5.2f} mg/L/h  early mu {m[0]:.4f}  T {df.temp_c[12:].mean():.2f}  Tmax {df.temp_c.rolling(6).mean().max():.2f}")
