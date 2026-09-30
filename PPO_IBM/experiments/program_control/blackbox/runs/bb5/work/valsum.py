import json, statistics as st, sys
import numpy as np
pref=sys.argv[1] if len(sys.argv)>1 else 'val_'
g={}
for line in open('../trials/results.jsonl'):
    r=json.loads(line); lab=r['batch'].split('_',1)[1]
    if not lab.startswith(pref): continue
    g.setdefault(r['inoculum'],[]).append((r['total_harvested_mg'],r['culture_lost'],r.get('error')))
allv=[]
print('inoc   n  median     p25     min     max  lost errors')
for k in sorted(g):
    v=[x[0] for x in g[k]]; allv+=v
    print('%5d %3d %7.0f %7.0f %7.0f %7.0f %4d %d'%(k,len(v),np.median(v),np.percentile(v,25),min(v),max(v),sum(x[1] for x in g[k]),sum(1 for x in g[k] if x[2])))
print('all  %3d median %.0f p25 %.0f lost %d'%(len(allv),np.median(allv),np.percentile(allv,25),sum(x[1] for k in g for x in g[k])))
typ=[x[0] for k in g if 100<=k<=400 for x in g[k]]
if typ: print('100-400 n=%d median %.0f p25 %.0f'%(len(typ),np.median(typ),np.percentile(typ,25)))
