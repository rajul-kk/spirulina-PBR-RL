import json, sys, pandas as pd, numpy as np
R = [json.loads(l) for l in open('../trials/results.jsonl')]
pats = sys.argv[1].split(',')
rows = []
for r in R:
    if not any(p in r['batch'] for p in pats): continue
    df = pd.read_csv(f"../trials/{r['batch']}.csv")
    x = np.array([h['lab_dry_weight_mg_per_L'] for h in r['harvests']]); t = np.array([h['hour'] for h in r['harvests']])
    mu = np.polyfit(t[:3], np.log(x[:3]), 1)[0]
    hold = df[(df.hour >= 48) & (df.hour < 96)]
    rows.append(dict(b=r['batch'], inoc=r['inoculum'], tot=r['total_harvested_mg'] / 1000, lost=r['culture_lost'], mu=mu, T=df.temp_c[12:].mean(),
                     Th=hold.temp_c.mean(), Lh=hold.light_umol.mean(), T0=df.temp_c[:3].mean(),
                     hold_mg=sum(h['harvested_mg'] for h in r['harvests'] if 48 <= h['hour'] <= 96) / 1000))
P = pd.DataFrame(rows); print(P.round(3).to_string())
if len(P) > 3:
    A = np.c_[np.ones(len(P)), P.mu]; c = np.linalg.lstsq(A, P.tot, rcond=None)[0]; P['resid'] = P.tot - A @ c
    print('tot ~ mu fit', c.round(2)); print(P.groupby(P.b.str.split('_').str[1:].str.join('_'))[['tot', 'resid', 'mu', 'Th', 'Lh', 'hold_mg']].mean().round(3))
