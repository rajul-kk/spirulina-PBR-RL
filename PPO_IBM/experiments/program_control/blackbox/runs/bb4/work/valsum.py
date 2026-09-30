import json, sys, numpy as np
pat=sys.argv[1] if len(sys.argv)>1 else 'val_'
R=[json.loads(l) for l in open('../trials/results.jsonl')]
R=[r for r in R if pat in r['batch']]
def bucket(i):
    for lo,hi in [(0,45),(45,80),(80,150),(150,250),(250,350),(350,600),(600,1500),(1500,3500),(3500,1e9)]:
        if lo<=i<hi: return (lo,hi)
B={}
for r in R: B.setdefault(bucket(r['inoculum']),[]).append(r)
def s(rs):
    v=np.array([r['total_harvested_mg'] for r in rs]); lost=sum(r['culture_lost'] for r in rs); err=sum(1 for r in rs if r.get('error'))
    return '%3d  median %7.0f  p25 %7.0f  min %7.0f  max %7.0f  lost %d err %d'%(len(v),np.median(v),np.percentile(v,25),v.min(),v.max(),lost,err)
for k in sorted(B): print('%5d-%-6d'%k, s(B[k]))
print('ALL        ', s(R))
rr=[r for r in R if 'rand' in r['batch']]
if rr: print('random-only', s(rr)); print(sorted(r['inoculum'] for r in rr))
