import json, sys, statistics as st
T='../trials/'
groups={}
for line in open(T+'results.jsonl'):
    r=json.loads(line)
    lab=r['batch'].split('_',1)[1]
    X96=[h['lab_dry_weight_mg_per_L'] for h in r['harvests'] if h['hour']==84][0]
    groups.setdefault(lab,[]).append((r['total_harvested_mg'],X96,r['culture_lost']))
for g in sys.argv[1:]:
    v=groups.get(g,[])
    tot=sorted(x[0] for x in v)
    x=[a[1] for a in v]
    n=len(tot)
    print('%-16s n=%2d mean %6.0f median %6.0f p25 %6.0f min %6.0f sd %5.0f | X84 mean %4.0f sd %3.0f lost %d'%(g,n,st.mean(tot),st.median(tot),tot[int(0.25*(n-1))],tot[0],st.pstdev(tot),st.mean(x),st.pstdev(x),sum(a[2] for a in v)))
