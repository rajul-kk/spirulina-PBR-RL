import sys, csv, json, glob, os
T = os.path.join(os.path.dirname(__file__), '..', 'trials')
res = {}
for l in open(os.path.join(T, 'results.jsonl')):
    r = json.loads(l); res[r['batch']] = r
step = int(sys.argv[2]) if len(sys.argv) > 2 else 12
for b in sorted(res):
    if sys.argv[1] not in b: continue
    r = res[b]
    print('==', b, 'inoc', r['inoculum'], 'tot', r['total_harvested_mg'], 'lost', r['culture_lost'], r.get('error'))
    dw = {h['hour']: (h['lab_dry_weight_mg_per_L'], h['harvested_mg']) for h in r['harvests']}
    rows = list(csv.DictReader(open(os.path.join(T, b + '.csv'))))
    print('hr   ntu     ph    pumpL   cond    temp   stir light hf  | DW  harv')
    for row in rows:
        h = float(row['hour'])
        if h % step == 0 or h in dw:
            d = dw.get(h, ('', ''))
            print(f"{h:5.0f} {float(row['turbidity_ntu']):7.1f} {float(row['ph']):5.2f} {float(row['pump_L']):6.2f} {float(row['conductivity']):7.0f} {float(row['temp_c']):5.2f} {float(row['stir_rpm']):5.0f} {float(row['light_umol']):5.0f} {float(row['harvest_frac']):.2f} | {d[0]} {d[1]}")
