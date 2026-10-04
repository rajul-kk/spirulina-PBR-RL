import numpy as np
from mysim import *
def ss(ca, Ti, Caf):
    rA = Caf-ca; k = rA/ca; T = 8750/np.log(7.2e10/k)
    Tc = T - ((Ti-T)+209.2050*rA)/2.0920502
    return T, Tc
for ca in [0.86,0.88,0.90]:
    for Ti in [348.5,351.5]:
        for Caf in [0.98,1.02]:
            T,Tc=ss(ca,Ti,Caf); print(ca,Ti,Caf,"T=%.2f Tc=%.2f"%(T,Tc))
# jacobian eig at nominal
T,Tc=ss(0.88,350,1.0); x=np.array([0.88,T]); J=np.zeros((2,2))
for i in range(2):
    d=np.zeros(2); d[i]=1e-6; J[:,i]=(rhs(x+d,Tc,350,1)-rhs(x-d,Tc,350,1))/2e-6
print("J",J,"eig",np.linalg.eigvals(J))
# bang-bang response from ss at 0.88
for u in [295,302]:
    xx=x.copy(); out=[]
    for k in range(15):
        xx=step(xx,u,350,1); out.append(round(xx[0],4))
    print(u,out)
# ignition check: max Tc, hot feed, starting hot
for x0 in [(0.85,326),(0.91,326)]:
    xx=np.array(x0,float); tm=0
    for k in range(120): xx=step(xx,302,351.5,1.02); tm=max(tm,xx[1])
    print("Tc=302 hold from",x0,"Tmax",tm, xx)
