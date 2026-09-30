import numpy as np, os
import fitgrowth as fg
th0=np.load(os.path.join(os.path.dirname(__file__),'th.npy'))
V=20
def run(inoc, target, E, s=1.0, dt=0.5):
    th=th0.copy(); th[0]*=s
    Lf=lambda X: min(2000,700+5*X)
    X=0.6*inoc; tot=0
    for k in range(11):
        Xs=X
        for _ in range(int(12/dt)): Xs*=np.exp(fg.mu(Xs,Lf(Xs),th)*dt)
        hn=k+1
        F=E.get(hn); 
        if F is None: F=min(0.5,max(0,1-target/Xs))
        tot+=F*Xs*V; X=Xs*(1-F)
    return tot
inocs=[30,100,200,300,1000]
for s in [0.75,1.0]:
  for target in [400,550,700]:
    for E in [{10:.5,11:.5},{9:.5,10:.5,11:.5},{8:.5,9:.5,10:.5,11:.5},{9:.35,10:.5,11:.5}]:
        print(s,target,sorted(E.items()),' '.join('%6.0f'%run(i,target,E,s) for i in inocs))
