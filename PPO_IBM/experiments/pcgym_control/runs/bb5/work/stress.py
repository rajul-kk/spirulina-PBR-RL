import numpy as np, localsim
def scen(rng):
    c1=rng.integers(15,60); c2=rng.integers(c1+15,105); vals=rng.uniform(0.84,0.92,3)
    sp=np.r_[np.full(c1,vals[0]),np.full(c2-c1,vals[1]),np.full(120-c2,vals[2])]
    caf=np.ones(120); tf=np.full(120,350.)
    for _ in range(3):
        k=rng.integers(0,120); caf[k:]=rng.uniform(0.92,1.08)
        k=rng.integers(0,120); tf[k:]=rng.uniform(345,355)
    return sp,caf,tf,np.array([rng.uniform(0.8,0.95),rng.uniform(315,330)])
localsim.scenario=scen
if __name__=="__main__":
  c,tm=localsim.run('controller',None,n=30,seed=5,mismatch=0.05)
  print('stress: mean %.3f median %.3f max %.3f Tmax %.1f runaways %d'%(c.mean(),np.median(c),c.max(),tm.max(),(tm>335).sum()))
