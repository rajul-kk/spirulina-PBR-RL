import json, collections, sys
T='../trials/'
pat=sys.argv[1]
edges=[0,50,100,200,300,450,600,800,1000,1250,1500,1800,2100,2500,3500]
b=collections.defaultdict(list)
for line in open(T+'results.jsonl'):
    r=json.loads(line)
    if pat not in r['controller']: continue
    hs=r['harvests']
    for i in range(len(hs)-1):
        if hs[i]['harvested_mg']>0: continue
        x0=hs[i]['lab_dry_weight_mg_per_L']; x1=hs[i+1]['lab_dry_weight_mg_per_L']
        xm=(x0+x1)/2
        for k in range(len(edges)-1):
            if edges[k]<=xm<edges[k+1]: b[k].append((x1-x0)/12)
for k in sorted(b):
    v=b[k]; print('%5d-%5d n=%2d P=%.2f mg/L/h  mu=%.4f'%(edges[k],edges[k+1],len(v),sum(v)/len(v), sum(v)/len(v)/((edges[k]+edges[k+1])/2)))
