import sys, csv, json, os
T = os.path.join(os.path.dirname(__file__), '..', 'trials')
pat = sys.argv[1] if len(sys.argv)>1 else ''
for l in open(os.path.join(T, 'results.jsonl')):
    r = json.loads(l)
    if pat not in r['batch']: continue
    rows = {float(x['hour']): x for x in csv.DictReader(open(os.path.join(T, r['batch'] + '.csv')))}
    out = []
    for h in r['harvests']:
        x = rows.get(h['hour'])
        if x is None: continue
        # average ntu over hour-1..hour to reduce noise
        ntu = sum(float(rows[k]['turbidity_ntu']) for k in (h['hour']-1, h['hour']) if k in rows)/2
        out.append('%d:%.2f' % (h['hour'], ntu/h['lab_dry_weight_mg_per_L']/(250/300)))
    print(r['batch'][:22].ljust(22), 'stir', rows[0.0]['stir_rpm'], ' '.join(out))
