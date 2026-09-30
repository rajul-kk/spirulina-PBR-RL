import sys, csv, json, os, numpy as np
T = os.path.join(os.path.dirname(__file__), '..', 'trials')
stir = float(sys.argv[1]) if len(sys.argv)>1 else 50
D=[]
for l in open(os.path.join(T, 'results.jsonl')):
    r = json.loads(l)
    rows = {float(x['hour']): x for x in csv.DictReader(open(os.path.join(T, r['batch'] + '.csv')))}
    if abs(float(rows[0.0]['stir_rpm'])-stir)>1: continue
    for h in r['harvests']:
        hh=h['hour']
        ntu = np.mean([float(rows[k]['turbidity_ntu']) for k in (hh-1, hh) if k in rows])
        D.append((hh, h['lab_dry_weight_mg_per_L'], ntu))
D=np.array(D); t,dw,ntu=D.T
best=None
for S in [200,300,400,500,600,700,800,1000,1500,1e9]:
  for a in np.linspace(0,0.006,31):
   for g in np.linspace(0.6,1.2,31):
      c=np.maximum(0.3,1-a*t)
      pred=S*(1-np.exp(-g*c*dw/S))
      e=np.mean((np.log(pred)-np.log(ntu))**2)
      if best is None or e<best[0]: best=(e,S,a,g)
e,S,a,g=best
print(len(D),'rms %.3f S=%g a=%.4f g=%.3f'%(np.sqrt(e),S,a,g))
c=np.maximum(0.3,1-a*t); pred=S*(1-np.exp(-g*c*dw/S))
res=np.log(ntu/pred)
for lo,hi in [(0,100),(100,200),(200,400),(400,700),(700,3000)]:
    m=(dw>=lo)&(dw<hi)
    if m.sum(): print(lo,hi,m.sum(),'mean res %.3f sd %.3f'%(res[m].mean(),res[m].std()))
for lo,hi in [(0,30),(30,60),(60,90),(90,150)]:
    m=(t>=lo)&(t<hi); print('t',lo,hi,'mean res %.3f'%res[m].mean())
