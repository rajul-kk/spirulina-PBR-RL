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
for S in [600,800,1000,1200,1500,2000,3000,1e9]:
  for a in np.linspace(0,0.006,61):
    for tau in [0]:
      c=np.maximum(0.3,1-a*t)
      pred=S*(1-np.exp(-0.8333*c*dw/S))
      e=np.mean((np.log(pred)-np.log(ntu))**2)
      if best is None or e<best[0]: best=(e,S,a)
print(len(D),'best rms log err %.3f S=%g a=%.4f'%(np.sqrt(best[0]),best[1],best[2]))
