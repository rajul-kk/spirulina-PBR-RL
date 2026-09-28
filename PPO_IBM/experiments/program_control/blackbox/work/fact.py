import json, pandas as pd, numpy as np, re
R = [json.loads(l) for l in open('../trials/results.jsonl')]
rows = []
for r in R:
    if 'E11' not in r['batch']: continue
    m = re.search(r'L(\d+)_s(\d+)_X(\d+)_d(\d)', r['batch'])
    df = pd.read_csv(f"../trials/{r['batch']}.csv")
    x = np.array([h['lab_dry_weight_mg_per_L'] for h in r['harvests']]); t = np.array([h['hour'] for h in r['harvests']])
    rows.append(dict(L=int(m[1]), s=int(m[2]), X=int(m[3]), d=int(m[4]), tot=r['total_harvested_mg'] / 1000,
                     mu=np.polyfit(t[:3], np.log(x[:3]), 1)[0], T=df.temp_c[12:].mean(), Tmax=df.temp_c.rolling(6).mean().max(), Lm=df.light_umol[12:].mean()))
P = pd.DataFrame(rows); print(P.round(3).to_string())
Z = np.c_[np.ones(16), (P.L == 2000) * 2 - 1, (P.s == 80) * 2 - 1, (P.X == 350) * 2 - 1, (P.d == 4) * 2 - 1]
c = np.linalg.lstsq(Z, P.tot, rcond=None)[0]; res = P.tot - Z @ c
print('effects (high-low, g): L2000', 2 * c[1], 's80', 2 * c[2], 'X350', 2 * c[3], 'd4', 2 * c[4], 'resid sd', res.std() * np.sqrt(16 / 11))
Z2 = np.c_[Z, P.mu - P.mu.mean()]; c2 = np.linalg.lstsq(Z2, P.tot, rcond=None)[0]; res2 = P.tot - Z2 @ c2
print('with mu covariate:', (2 * c2[1:5]).round(2), 'mu coef', c2[5].round(1), 'resid sd', (res2.std() * np.sqrt(16 / 10)).round(2))
