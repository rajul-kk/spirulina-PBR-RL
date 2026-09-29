"""Per batch: early growth rate (fit ln DW vs t over assays with DW<700 and no harvest), mean temp, light, stir."""
import json, csv, math, sys
import numpy as np
TR = '../trials/'
sel = sys.argv[1] if len(sys.argv) > 1 else ''
for l in open(TR + 'results.jsonl'):
    r = json.loads(l)
    if sel not in r['batch']: continue
    rows = list(csv.DictReader(open(TR + r['batch'] + '.csv')))
    H = r['harvests']
    pts = [(h['hour'], h['lab_dry_weight_mg_per_L']) for h in H if h['hour'] <= 96 and h['lab_dry_weight_mg_per_L'] < 700]
    if len(pts) < 3: continue
    t = np.array([p[0] for p in pts]); y = np.log([p[1] for p in pts])
    mu = np.polyfit(t, y, 1)[0]
    seg = [x for x in rows if float(x['hour']) <= t[-1]]
    T = np.mean([float(x['temp_c']) for x in seg]); L = np.mean([float(x['light_umol']) for x in seg]); S = np.mean([float(x['stir_rpm']) for x in seg])
    print(f"{r['batch']:<22} inoc {r['inoculum']:>5} S {S:5.0f} L {L:6.0f} T {T:5.1f} mu {mu:.4f} tot {r['total_harvested_mg']:8.0f}")
