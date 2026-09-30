import json,csv,collections
T='../trials/'
bins=collections.defaultdict(list)
for line in open(T+'results.jsonl'):
    r=json.loads(line)
    rows={float(x['hour']):x for x in csv.DictReader(open(T+r['batch']+'.csv'))}
    for h in r['harvests']:
        X=h['lab_dry_weight_mg_per_L']; hr=h['hour']
        # average turbidity in hour before harvest
        tb=float(rows[hr-1]['turbidity_ntu'])
        if X>1100 or X<15: continue
        bins[hr].append(tb/X)
for hr in sorted(bins):
    v=bins[hr]; v.sort()
    print(hr, len(v), 'median %.3f  min %.3f max %.3f'%(v[len(v)//2],v[0],v[-1]))
