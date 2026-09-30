import csv, json, os, numpy as np
from scipy.optimize import least_squares
T = os.path.join(os.path.dirname(__file__), '..', 'trials')
D=[]
for l in open(os.path.join(T, 'results.jsonl')):
    r = json.loads(l)
    rows = {float(x['hour']): x for x in csv.DictReader(open(os.path.join(T, r['batch'] + '.csv')))}
    if abs(float(rows[0.0]['stir_rpm'])-50)>1: continue
    for h in r['harvests']:
        hh=h['hour']
        ntu = np.mean([float(rows[k]['turbidity_ntu']) for k in (hh-2, hh-1) if k in rows])
        D.append((hh-1.5, h['lab_dry_weight_mg_per_L'], ntu))
D=np.array(D); t,dw,ntu=D.T
def pred(p,t,dw):
    g,S,cmax,tau,sh=p
    c=1-cmax*(1-np.exp(-np.maximum(t-sh,0)/tau))
    x=g*c*dw
    return S*(1-np.exp(-x/S))
def res(p): return np.log(pred(p,t,dw)/ntu)
for p0 in [[0.83,800,0.3,50,0],[0.83,500,0.5,100,0],[0.83,2000,0.4,30,10]]:
    r=least_squares(res,p0,bounds=([0.5,100,0,1,0],[1.2,1e5,0.9,1000,48]))
    print(np.round(r.x,4),'rms %.4f'%np.sqrt(np.mean(r.fun**2)))
p=r.x; e=res(p)
for lo,hi in [(0,100),(100,200),(200,400),(400,700),(700,3000)]:
    m=(dw>=lo)&(dw<hi); print(lo,hi,m.sum(),'%.3f'%e[m].mean())
for lo,hi in [(0,30),(30,60),(60,90),(90,115),(115,150)]:
    m=(t>=lo)&(t<hi); print('t',lo,hi,'%.3f'%e[m].mean())
