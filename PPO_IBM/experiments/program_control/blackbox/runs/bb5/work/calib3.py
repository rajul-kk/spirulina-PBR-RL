import json,csv,math
import numpy as np
T='../trials/'
pts=[]
for line in open(T+'results.jsonl'):
    r=json.loads(line)
    rows={round(float(x['hour'])):x for x in csv.DictReader(open(T+r['batch']+'.csv'))}
    for hh in r['harvests']:
        X=hh['lab_dry_weight_mg_per_L']; hr=round(hh['hour'])
        tb=float(rows[hr-1]['turbidity_ntu'])
        if tb>940: continue
        pts.append((X,hr,tb))
pts=np.array(pts)
knots=np.array([0,50,150,300,550,850,1200,1700,2300,3000])
best=None
for c in np.linspace(0,0.004,17):
    fac=1-c*(pts[:,1]-48)
    y=pts[:,2]/fac
    # fit base values at knots by least squares on linear interpolation basis
    A=np.zeros((len(pts),len(knots)))
    for i,x in enumerate(pts[:,0]):
        j=np.searchsorted(knots,x)-1; j=min(max(j,0),len(knots)-2)
        a=(x-knots[j])/(knots[j+1]-knots[j]); A[i,j]=1-a; A[i,j+1]=a
    w=1/np.maximum(y,20)
    sol,*_=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)
    pred=(A@sol)*fac
    err=np.sqrt(np.mean(np.log(pts[:,2]/np.maximum(pred,1))**2))
    if best is None or err<best[0]: best=(err,c,sol)
err,c,sol=best
print('c=%.4f rms log err %.3f'%(c,err)); print('knots',knots.tolist()); print('base',[round(v,1) for v in sol])
# inverse error: estimate X from NTU and compare
fac=1-c*(pts[:,1]-48)
est=np.interp(pts[:,2]/fac, sol, knots)
rel=est/pts[:,0]
for lo,hi in [(0,200),(200,400),(400,700),(700,1000),(1000,1500),(1500,4000)]:
    m=(pts[:,0]>=lo)&(pts[:,0]<hi)
    if m.sum(): print(lo,hi,m.sum(),'median est/true %.2f  p10 %.2f p90 %.2f'%(np.median(rel[m]),np.percentile(rel[m],10),np.percentile(rel[m],90)))
