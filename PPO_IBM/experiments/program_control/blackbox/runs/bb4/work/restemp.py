import numpy as np, os
from fitgrowth import load, resid
th=np.load(os.path.join(os.path.dirname(__file__),'th.npy'))
I=load(50); r=resid(np.log(th),I)
Tm=np.array([i[5].mean() for i in I]); ok=np.isfinite(r)
for lo,hi in [(30,33),(33,35),(35,37),(37,39),(39,42)]:
    m=ok&(Tm>=lo)&(Tm<hi); print(lo,hi,m.sum(),'%.3f'%r[m].mean())
print(np.corrcoef(Tm[ok],r[ok]))
