"""Pool 12h assay intervals: growth rate & productivity vs density, per (stir, light) group.
Accounts for harvest: X_after = X_before*(1-f_eff), f_eff from pump volume change / volume(~20.9L)."""
import json, csv, math, sys, collections
TR = '../trials/'
V = 20.9
out = collections.defaultdict(list)
for l in open(TR + 'results.jsonl'):
    r = json.loads(l)
    rows = list(csv.DictReader(open(TR + r['batch'] + '.csv')))
    H = r['harvests']
    pump = {float(x['hour']): float(x['pump_L']) for x in rows}
    for k in range(len(H) - 1):
        t0, t1 = H[k]['hour'], H[k+1]['hour']
        seg = [x for x in rows if t0 <= float(x['hour']) < t1]
        L = sum(float(x['light_umol']) for x in seg) / len(seg)
        S = sum(float(x['stir_rpm']) for x in seg) / len(seg)
        T = sum(float(x['temp_c']) for x in seg) / len(seg)
        f = (pump.get(t0 + 1, 0) - pump.get(t0 - 1, 0)) / V
        x0 = H[k]['lab_dry_weight_mg_per_L'] * (1 - max(0, f))
        x1 = H[k+1]['lab_dry_weight_mg_per_L']
        mu = math.log(x1 / x0) / 12
        key = (round(S, -1), round(L, -2))
        out[key].append((x0, x1, mu, (x1 - x0) / 12, T, f))
bins = [0, 50, 100, 200, 300, 450, 600, 800, 3000]
for key in sorted(out):
    print('stir,light', key, 'n', len(out[key]))
    for a, b in zip(bins, bins[1:]):
        v = [o for o in out[key] if a <= math.sqrt(o[0]*o[1]) < b]
        if not v: continue
        mu = sum(o[2] for o in v) / len(v); pr = sum(o[3] for o in v)/len(v)
        print(f'   X {a:>4}-{b:<4} n{len(v):>3} mu {mu:.4f} dX/dt {pr:6.2f}')
