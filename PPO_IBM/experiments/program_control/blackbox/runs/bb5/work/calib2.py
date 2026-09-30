import json,csv,collections
T='../trials/'
tab=collections.defaultdict(list)
xe=[0,100,200,400,700,1000,1400,2000,4000]
for line in open(T+'results.jsonl'):
    r=json.loads(line)
    rows={round(float(x['hour'])):x for x in csv.DictReader(open(T+r['batch']+'.csv'))}
    for hh in r['harvests']:
        X=hh['lab_dry_weight_mg_per_L']; hr=round(hh['hour'])
        tb=float(rows[hr-1]['turbidity_ntu'])
        xi=max(i for i in range(len(xe)-1) if X>=xe[i])
        tab[(xi,hr//24)].append(tb/X)
print('X-bin \ day', ' '.join('  d%d  '%d for d in range(6)))
for xi in range(len(xe)-1):
    s=[]
    for d in range(6):
        v=tab.get((xi,d),[])
        s.append('%.2f(%2d)'%(sorted(v)[len(v)//2],len(v)) if v else '   -    ')
    print('%4d-%4d'%(xe[xi],xe[xi+1]),' '.join(s))
