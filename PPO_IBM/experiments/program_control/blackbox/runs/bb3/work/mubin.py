import json,csv,math,sys
import numpy as np
R=[json.loads(l) for l in open('../trials/results.jsonl')]
D=[]
Vs=[]
for r in R:
    if r['culture_lost']: continue
    rows=list(csv.DictReader(open('../trials/%s.csv'%r['batch'])))
    H=r['harvests']
    for k in range(len(H)):
        t0=int(H[k]['hour']); seg=rows[t0-12:t0]
        fr=np.mean([float(x['harvest_frac']) for x in seg])
        if fr>0.02: Vs.append(H[k]['harvested_mg']/(H[k]['lab_dry_weight_mg_per_L']*fr))
    for k in range(len(H)-1):
        d0=H[k]['lab_dry_weight_mg_per_L']; d1=H[k+1]['lab_dry_weight_mg_per_L']
        t0=int(H[k]['hour']); t1=int(H[k+1]['hour'])
        seg=rows[t0:t1]
        f=H[k]['harvested_mg']/(d0*20.0)
        if f>0.45: continue
        mu=math.log(d1/(d0*(1-f)))/12
        L=np.mean([float(x['light_umol']) for x in seg]); T=np.mean([float(x['temp_c']) for x in seg]); S=np.mean([float(x['stir_rpm']) for x in seg])
        D.append((d0*(1-f),mu,L,T,S,t0))
print('V est median %.1f (n=%d, iqr %.1f-%.1f)'%(np.median(Vs),len(Vs),np.percentile(Vs,25),np.percentile(Vs,75)))
D=np.array(D)
bins=[0,40,80,150,250,400,600,850,1100,1500]
lb=[(0,450),(450,900),(900,1400),(1400,2100)]
print('rows: X bin; cols light bins',lb)
for i in range(len(bins)-1):
    s=(D[:,0]>=bins[i])&(D[:,0]<bins[i+1])
    out=[]
    for lo,hi in lb:
        q=s&(D[:,2]>=lo)&(D[:,2]<hi)
        out.append('%.4f(%2d)'%(D[q,1].mean(),q.sum()) if q.sum() else '    -     ')
    print('%5d-%5d'%(bins[i],bins[i+1]),' '.join(out))
np.save('mu_data.npy',D)
