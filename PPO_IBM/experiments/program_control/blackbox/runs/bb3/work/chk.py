import numpy as np,sys
r=float(sys.argv[1]); K=float(sys.argv[2]); V=20.
def grow(X,h=12): return K/(1+(K/X-1)*np.exp(-r*h))
for X0,f in [(1000,0.1),(1000,0.2),(1000,0.05),(150,0.1),(150,0.0),(22,0.0)]:
    X=X0; tot=0; tr=[]
    for k in range(11):
        tr.append(int(X)); tot+=f*X*V; X=grow(X*(1-f))
    print(X0,f,int(tot),tr)
