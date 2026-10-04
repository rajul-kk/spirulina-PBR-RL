import numpy as np, localsim, warnings; warnings.filterwarnings('ignore')
from stress import scen
localsim.scenario=scen
import sys
for mod in sys.argv[1:]:
    c,tm=localsim.run(mod,None,n=30,seed=5,mismatch=0.05)
    print(mod,'per-batch Tmax>333:',[(i,round(t,1),round(cc,2)) for i,(t,cc) in enumerate(zip(tm,c)) if t>333 or not np.isfinite(cc)])
