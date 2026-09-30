import sys, csv, json
T = '../trials/'
def load(b):
    with open(T+b+'.csv') as f:
        return list(csv.DictReader(f))
res = {}
for line in open(T+'results.jsonl'):
    r = json.loads(line); res[r['batch']] = r
for b in sys.argv[1:]:
    rows = load(b); r = res[b]
    print(b, 'inoc', r['inoculum'], 'total', r['total_harvested_mg'], 'lost', r['culture_lost'])
    lab = {h['hour']: (h['lab_dry_weight_mg_per_L'], h['harvested_mg']) for h in r['harvests']}
    step = int(sys.argv[0] and 6)
    for row in rows:
        h = float(row['hour'])
        if h % 6 == 0 or h in lab:
            l = lab.get(h, ('', ''))
            print(' %5.0f turb %7.1f ph %5.2f pump %7.2f cond %8.0f T %5.2f lux %6.0f | %4.0f %5.0f %.3f | lab %s harv %s' % (
                h, float(row['turbidity_ntu']), float(row['ph']), float(row['pump_L']), float(row['conductivity']),
                float(row['temp_c']), float(row['lux']), float(row['stir_rpm']), float(row['light_umol']), float(row['harvest_frac']), l[0], l[1]))
