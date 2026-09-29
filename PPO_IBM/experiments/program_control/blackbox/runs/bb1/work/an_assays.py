"""Summarise lab assays per batch: DW series, interval growth rates, mean temp per interval."""
import json, sys, csv, math
TR = '../trials/'
sel = sys.argv[1] if len(sys.argv) > 1 else ''
for l in open(TR + 'results.jsonl'):
    r = json.loads(l)
    if sel not in r['batch']: continue
    rows = list(csv.DictReader(open(TR + r['batch'] + '.csv')))
    H = r['harvests']
    dw = [h['lab_dry_weight_mg_per_L'] for h in H]
    temps = []
    for k in range(len(H)):
        t0 = 0 if k == 0 else H[k-1]['hour']; t1 = H[k]['hour']
        ts = [float(x['temp_c']) for x in rows if t0 <= float(x['hour']) < t1]
        temps.append(sum(ts)/max(1, len(ts)))
    tmax = max(float(x['temp_c']) for x in rows)
    print(f"{r['batch']:<22} inoc {r['inoculum']:>5} tot {r['total_harvested_mg']:>8.0f} lost {r['culture_lost']}")
    print('   DW  ', ' '.join(f'{d:6.0f}' for d in dw))
    print('   T   ', ' '.join(f'{t:6.1f}' for t in temps), f' Tmax {tmax:.1f}')
    tu = [float(x['turbidity_ntu']) for x in rows if float(x['hour']) % 12 == 11]
    print('   NTU ', ' '.join(f'{t:6.0f}' for t in tu))
