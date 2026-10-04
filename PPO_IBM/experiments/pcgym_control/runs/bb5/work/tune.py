import sys, json, numpy as np
from localsim import run
mm=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
for s in sys.argv[1].split(';'):
    p=json.loads(s); c,tm=run('controller',p,n=30,seed=1,mismatch=mm)
    print(s,'mean %.3f median %.3f max %.3f Tmax %.1f'%(c.mean(),np.median(c),c.max(),tm.max()),flush=True)
