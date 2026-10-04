"""Local test bench built from my identified model (not the plant)."""
import numpy as np, importlib, sys, time
sys.path.insert(0,'.')
from simfit import step as fstep, P as P0
def scenario(rng):
    sps=[]; t=0
    c1=rng.integers(15,60); c2=rng.integers(c1+15,105)
    vals=rng.uniform(0.85,0.90,3)
    sp=np.r_[np.full(c1,vals[0]),np.full(c2-c1,vals[1]),np.full(120-c2,vals[2])]
    caf=np.ones(120); tf=np.full(120,350.)
    for _ in range(rng.integers(1,4)):
        k=rng.integers(0,120); caf[k:]=rng.uniform(0.97,1.03)
    for _ in range(rng.integers(1,4)):
        k=rng.integers(0,120); tf[k:]=rng.uniform(348.5,351.5)
    x0=np.array([rng.uniform(0.86,0.89),rng.uniform(320,326)])
    return sp,caf,tf,x0
def run(modname,params=None,n=20,seed=0,mismatch=0.0,verbose=False):
    mod=importlib.import_module(modname); rng=np.random.default_rng(seed); costs=[];tmax=[]
    for b in range(n):
        sp,caf,tf,x=scenario(rng)
        p=dict(P0)
        if mismatch:
            p['U']*=1+mismatch*rng.normal(); p['J']*=1+mismatch*rng.normal(); p['k0']*=1+3*mismatch*rng.normal()
        c=mod.Controller(params); e=0; Tm=0
        for k in range(120):
            obs=dict(t_min=k*13/60,Ca=x[0]+0.002*rng.normal(),T=x[1]+0.2*rng.normal(),Ca_sp=sp[k],_z=(x[0],x[1],caf[k],tf[k]))
            e+=((x[0]-sp[k])/0.01)**2; Tm=max(Tm,x[1])
            u=float(np.clip(c.act(obs),295,302)); p['Caf']=caf[k]; p['Tf']=tf[k]
            x=fstep(x,u,p,n=10)
        costs.append(e/120); tmax.append(Tm)
    return np.array(costs),np.array(tmax)
if __name__=='__main__':
    t=time.time(); c,tm=run(sys.argv[1] if len(sys.argv)>1 else 'controller',n=int(sys.argv[2]) if len(sys.argv)>2 else 10)
    print('mean',c.mean().round(3),'median',np.median(c).round(3),'max',c.max().round(3),'Tmax',tm.max().round(1),'time',round(time.time()-t,1))
