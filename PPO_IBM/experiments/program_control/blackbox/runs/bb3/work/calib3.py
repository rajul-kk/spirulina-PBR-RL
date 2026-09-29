import json,csv,sys
import numpy as np
R=[json.loads(l) for l in open('../trials/results.jsonl')]
A=[]
for r in R:
    rows=list(csv.DictReader(open('../trials/%s.csv'%r['batch'])))
    for h in r['harvests']:
        t=int(h['hour']); dw=h['lab_dry_weight_mg_per_L']
        ntu=float(rows[t-1]['turbidity_ntu']); st=np.mean([float(x['stir_rpm']) for x in rows[:t]])
        if ntu>950: continue
        A.append((ntu,t,st,dw))
A=np.array(A)
for sv in [50,120]:
    m=np.abs(A[:,2]-sv)<3
    X=np.c_[np.ones(m.sum()),np.log(A[m,0]),A[m,1]]
    y=np.log(A[m,3]); co,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@co
    print(sv,m.sum(),'coef',co,'rms %.3f'%res.std())
