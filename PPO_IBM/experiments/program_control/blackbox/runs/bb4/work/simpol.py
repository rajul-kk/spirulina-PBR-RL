import numpy as np, os, itertools
from fitgrowth import mu
th=np.load(os.path.join(os.path.dirname(__file__),'th.npy'))
V=20
def run(inoc, target, Lfun, endF, dt=0.25):
    X=0.6*inoc; tot=0
    for k in range(12):
        # decide F for harvest at end of interval k (hour 12(k+1)); perfect-info: simulate to end
        Xs=X
        for s in range(int(12/dt)):
            Xs*=np.exp(mu(Xs,Lfun(Xs),th)*dt)
        hn=k+1
        if hn<=11:
            F = endF.get(hn, None)
            if F is None: F=min(0.5,max(0,1-target/Xs))
        else: F=0
        tot+=F*Xs*V; X=Xs*(1-F)
    return tot
L1=lambda X: min(2000, 800+3*X)
for target in [300,400,500,600,700,900]:
  for endF in [{}, {11:.5},{10:.5,11:.5},{9:.5,10:.5,11:.5},{8:.5,9:.5,10:.5,11:.5}]:
    res=[run(i,target,L1,endF) for i in [30,100,200,300,1000,3000]]
    print(target, sorted(endF), ' '.join('%6.0f'%r for r in res))
