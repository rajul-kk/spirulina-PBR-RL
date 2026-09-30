import numpy as np, sys
from fitgrowth import load
for pat in sys.argv[1:]:
    I=load(50,pat)
    for b in sorted(set(i[0] for i in I)):
        rows=[i for i in I if i[0]==b]
        s=[]
        for (_,h,x0,x1,Ls,Ts) in rows:
            s.append('%3d L%4d T%4.1f x%4d->%4d P%5.1f'%(h,Ls.mean(),Ts.mean(),x0,x1,(x1-x0)/12))
        print(b); print('\n'.join(s))
        hi=[ (i[3]-i[2])/12 for i in rows if i[4].mean()>1700]; lo=[(i[3]-i[2])/12 for i in rows if i[4].mean()<1500]
        print('  mean P hiL %.2f loL %.2f'%(np.mean(hi),np.mean(lo)))
