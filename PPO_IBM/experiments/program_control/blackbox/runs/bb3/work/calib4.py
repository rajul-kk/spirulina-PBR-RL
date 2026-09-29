import json,csv
import numpy as np
R=[json.loads(l) for l in open('../trials/results.jsonl')]
A=[]
for r in R:
    rows=list(csv.DictReader(open('../trials/%s.csv'%r['batch'])))
    st=np.mean([float(x['stir_rpm']) for x in rows])
    if abs(st-50)>3: continue
    for h in r['harvests']:
        t=int(h['hour']); dw=h['lab_dry_weight_mg_per_L']
        ntu=np.mean([float(rows[i]['turbidity_ntu']) for i in (t-2,t-1)])
        if ntu>950: continue
        A.append((ntu,t,dw,r['inoculum']))
A=np.array(A); print('n',len(A))
ln=np.log(A[:,0]); y=np.log(A[:,2]); h=A[:,1]
for name,X in [('lin',np.c_[np.ones(len(A)),ln,h]),('quad',np.c_[np.ones(len(A)),ln,ln**2,h]),('quad+hx',np.c_[np.ones(len(A)),ln,ln**2,h,h*ln])]:
    co,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@co
    print(name,co,'rms %.3f'%res.std())
    for lo,hi in [(0,100),(100,300),(300,600),(600,1000)]:
        m=(A[:,0]>=lo)&(A[:,0]<hi); print('   ntu %d-%d n=%d bias %.3f rms %.3f'%(lo,hi,m.sum(),res[m].mean(),res[m].std()))
    d=A[:,3]>=3000; print('   dense-start bias %.3f'%res[d].mean())
