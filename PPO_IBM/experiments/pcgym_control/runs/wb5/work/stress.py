import numpy as np, warnings; warnings.filterwarnings('ignore')
import mysim
from controller import Controller
N=120
cases=[]
for sp in [(0.86,0.90,0.86),(0.90,0.86,0.86)]:
  for Ti in [348.5,351.5]:
    for Caf in [0.98,1.02]:
      for x0 in [(0.85,326.),(0.91,320.)]:
        cases.append((sp,Ti,Caf,x0))
orig=mysim.scenario
res=[]
for i,(sp,Ti,Caf,x0) in enumerate(cases):
    s=np.repeat(sp,[40,40,40]).astype(float)
    mysim.scenario=lambda seed,s=s,Ti=Ti,Caf=Caf,x0=x0: {"sp":s,"Ti":np.full(N,Ti),"Caf":np.full(N,Caf),"x0":np.array(x0),"noise_seed":seed}
    c,ra,tr=mysim.run(Controller(),i,trace=True)
    res.append(c); print(sp,Ti,Caf,x0,"cost %.3f Tmax %.1f runaway %s"%(c,tr[:,6].max(),ra))
print("mean",np.mean(res))
