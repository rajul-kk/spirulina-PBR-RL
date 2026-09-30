import json, csv, numpy as np, sys
R=[json.loads(l) for l in open('../trials/results.jsonl')]
groups=sys.argv[1:]
for g in groups:
    rs=[r for r in R if g in r['batch']]
    P=[];tot=[];X=[];T=[]
    for r in rs:
        H=r['harvests']; gsum=0; xs=[]
        for a,b in zip(H[:-1],H[1:]):
            F=a['harvested_mg']/(20*a['lab_dry_weight_mg_per_L'])
            gsum+=b['lab_dry_weight_mg_per_L']-a['lab_dry_weight_mg_per_L']*(1-F); xs.append(a['lab_dry_weight_mg_per_L'])
        P.append(gsum/120); tot.append(r['total_harvested_mg']); X.append(np.mean(xs))
        rows=list(csv.DictReader(open('../trials/%s.csv'%r['batch'])))
        T.append(np.mean([float(x['temp_c']) for x in rows[12:]]))
    print('%-8s n=%d prod %s mean %.2f | tot mean %.0f | X %.0f | T %.1f'%(g,len(rs),' '.join('%.1f'%p for p in P),np.mean(P),np.mean(tot),np.mean(X),np.mean(T)))
