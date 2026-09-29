import json,numpy as np
R=[json.loads(l) for l in open('../trials/results.jsonl')]
V=[r for r in R if '_val' in r['batch']]
def st(rs,name):
    if not rs: return
    t=np.array([r['total_harvested_mg'] for r in rs]); lost=sum(r['culture_lost'] for r in rs)
    print('%-28s n=%3d median %6.0f p25 %6.0f min %6.0f max %6.0f lost %d'%(name,len(rs),np.median(t),np.percentile(t,25),t.min(),t.max(),lost))
rnd=[r for r in V if '_valR' in r['batch']]
st(rnd,'random (valR+valR2)')
for lo,hi in [(0,100),(100,200),(200,400),(400,1000),(1000,3000),(3000,10000)]:
    st([r for r in rnd if lo<=r['inoculum']<hi],'  random inoc %d-%d'%(lo,hi))
for i in [30,100,200,400,1000,3000,5000]:
    st([r for r in V if r['batch'].endswith('_valI%d'%i)],'fixed inoc %d'%i)
st(V,'all validation')
print('all batches lost:',sum(r['culture_lost'] for r in R),'of',len(R))
