import sys, glob, numpy as np, json
from simfit import load
from controller import Controller
def score(params, pat='../trials/b*.csv', ahead=3):
    E1=[];E3=[]
    for fn in sorted(glob.glob(pat)):
        d=load(fn); c=Controller(params); n=len(d['Tc'])
        zs=[]
        for i in range(n):
            c._ekf((d['Ca'][i],d['T'][i])); c.u_prev=d['Tc'][i]; zs.append(c.z.copy())
        # k-step ahead prediction from filtered state at i using logged Tc
        from controller import _step
        for i in range(5,n-ahead):
            Ca,T,Caf,Tf=zs[i]
            for j in range(ahead):
                Ca,T=_step(Ca,T,d['Tc'][i+j],Caf,Tf,4)
                if j==0: E1.append(d['Ca'][i+1]-Ca)
            E3.append(d['Ca'][i+ahead]-Ca)
    E1=np.array(E1);E3=np.array(E3)
    return np.sqrt((E1**2).mean()), E1.mean(), np.sqrt((E3**2).mean()), E3.mean()
if __name__=='__main__':
    for s in sys.argv[1].split(';'):
        p=json.loads(s); print(s,'1-step rms %.5f bias %.5f | 3-step rms %.5f bias %.5f'%score(p))
