import json,csv
import numpy as np
R=[json.loads(l) for l in open('../trials/results.jsonl')]
A=[]
for r in R:
    rows=list(csv.DictReader(open('../trials/%s.csv'%r['batch'])))
    for h in r['harvests']:
        t=int(h['hour']); dw=h['lab_dry_weight_mg_per_L']
        ntu=float(rows[t-1]['turbidity_ntu']); st=float(rows[t-1]['stir_rpm'])
        if ntu>950: continue
        A.append((ntu,t,st,dw))
A=np.array(A)
m=A[:,2]==120
# model log(dw) = a + b log(ntu) + c t
X=np.c_[np.ones(m.sum()),np.log(A[m,0]),A[m,1]]
y=np.log(A[m,3]); co,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@co
print('coef',co,'rms',res.std())
X=np.c_[np.ones(m.sum()),np.log(A[m,0])]
co2,*_=np.linalg.lstsq(X,y,rcond=None); print('coef no t',co2,'rms',(y-X@co2).std())
for lo,hi in [(0,50),(50,150),(150,300),(300,500),(500,1000)]:
    s=m&(A[:,0]>=lo)&(A[:,0]<hi)
    print(lo,hi,s.sum(),'dw/ntu median %.2f'%np.median(A[s,3]/A[s,0]))
