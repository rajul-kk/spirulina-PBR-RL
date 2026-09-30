import sys, csv, json, os, numpy as np
T = os.path.join(os.path.dirname(__file__), '..', 'trials')
pat = sys.argv[1] if len(sys.argv)>1 else ''
V=20.0
D=[]
for l in open(os.path.join(T, 'results.jsonl')):
    r = json.loads(l)
    if pat not in r['batch']: continue
    rows = {float(x['hour']): x for x in csv.DictReader(open(os.path.join(T, r['batch'] + '.csv')))}
    if abs(float(rows[0.0]['stir_rpm'])-50)>1: continue
    H=r['harvests']
    for a,b in zip(H[:-1],H[1:]):
        F=a['harvested_mg']/(V*a['lab_dry_weight_mg_per_L'])
        x0=a['lab_dry_weight_mg_per_L']*(1-F); x1=b['lab_dry_weight_mg_per_L']
        L=np.mean([float(rows[k]['light_umol']) for k in np.arange(a['hour'],b['hour'])])
        Tc=np.mean([float(rows[k]['temp_c']) for k in np.arange(a['hour'],b['hour'])])
        D.append((a['hour'],x0,x1,(x1-x0)/12,np.log(x1/x0)/12,L,Tc))
D=np.array(D)
print('n',len(D))
for Llo,Lhi in [(0,600),(600,1000),(1000,1500),(1500,1900),(1900,2100)]:
  for lo,hi in [(0,50),(50,100),(100,200),(200,300),(300,450),(450,650),(650,900),(900,3000)]:
    xm=np.sqrt(D[:,1]*D[:,2])
    m=(xm>=lo)&(xm<hi)&(D[:,5]>=Llo)&(D[:,5]<Lhi)
    if m.sum()>=2: print('L %4d-%4d X %4d-%4d n=%2d prod %.2f mg/L/h (sd %.2f) mu %.4f'%(Llo,Lhi,lo,hi,m.sum(),D[m,3].mean(),D[m,3].std(),D[m,4].mean()))
