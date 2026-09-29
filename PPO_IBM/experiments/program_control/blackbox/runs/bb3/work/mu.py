import json,csv,sys,math
import numpy as np
R=[json.loads(l) for l in open('../trials/results.jsonl')]
pat=sys.argv[1] if len(sys.argv)>1 else ''
out=[]
for r in R:
    if pat not in r['batch'] or r['culture_lost']: continue
    rows=list(csv.DictReader(open('../trials/%s.csv'%r['batch'])))
    H=r['harvests']
    for k in range(len(H)-1):
        d0=H[k]['lab_dry_weight_mg_per_L']; d1=H[k+1]['lab_dry_weight_mg_per_L']
        # fraction actually harvested at k: harvested_mg/(d0*V), V ~ 20 L est via pump
        t0=int(H[k]['hour']); t1=int(H[k+1]['hour'])
        seg=rows[t0:t1]
        f=H[k]['harvested_mg']/(d0*20.0) if d0>0 else 0
        mu=math.log(d1/(d0*(1-f)))/(t1-t0)
        L=np.mean([float(x['light_umol']) for x in seg]); S=np.mean([float(x['stir_rpm']) for x in seg]); T=np.mean([float(x['temp_c']) for x in seg])
        out.append((r['batch'],t0,d0,f,mu,L,S,T))
        print('%-14s h%3d DW %6.0f f %.2f mu %.4f L %5.0f S %4.0f T %.1f'%out[-1])
