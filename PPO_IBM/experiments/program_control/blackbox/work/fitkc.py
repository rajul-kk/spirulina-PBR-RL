import json, importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location('c', 'ctrl_v1.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
R = [json.loads(l) for l in open('../trials/results.jsonl')]
data = []
for r in R:
    if not any(s in r['batch'] for s in ['E5_s50', 'E6', 'E7', 'E8']): continue
    df = pd.read_csv(f"../trials/{r['batch']}.csv")
    hrs = np.arange(0, 7200) * 0.02
    data.append((np.interp(hrs, df.hour, df.turbidity_ntu), df.stir_rpm.median(),
                 {round(h['hour']): h['lab_dry_weight_mg_per_L'] for h in r['harvests']}))
def score(kc, K):
    lr = []
    for ntu, stir, dw in data:
        c = 1.0
        for t in range(7200):
            X = -K * np.log(1 - min(ntu[t], 950) / 1000) / c
            c = max(0.4, c - kc * X * 0.02)
            if (t + 1) % 600 == 0 and round((t + 1) * 0.02) in dw: lr.append(np.log(X / dw[round((t + 1) * 0.02)]))
    lr = np.array(lr); return lr.mean(), lr.std()
for K in [1100, 1230, 1400]:
    for kc in [4.7e-6, 7e-6, 9e-6, 12e-6]:
        print(K, kc, np.round(score(kc, K), 3))
