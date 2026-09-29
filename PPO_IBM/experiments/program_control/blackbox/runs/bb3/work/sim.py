import numpy as np
p=np.load('fit_p2.npy')
def mu_f(X,L,S=50,scale=1.0):
    mumax,Ks,Ki,k,m,se,c=p
    Ib=L*(1-np.exp(-k*X))/(k*X)
    return scale*mumax*Ib/(Ks+Ib+Ib**2/Ki)*(1+se*(120-S)/70) - m - c*X/1000
def light(X,Lmax=1800):
    k=p[3]; return float(np.clip(253*k*X/(1-np.exp(-k*X)),300,Lmax))
def run(X0,hold,E,scale=1.0,Lmax=1800,hold96=None):
    X=X0; tot=0
    for step in range(7200):
        h=(step+1)*0.02
        X*=np.exp(mu_f(X,light(X,Lmax),50,scale)*0.02)
        if (step+1)%600==0 and h<143:
            if h>=E-1e-6: f=0.5
            else:
                hh=hold96 if (hold96 and h>=95) else hold
                f=min(0.5,max(0,1-hh/X))
            tot+=f*X*20; X*=(1-f)
    return tot
if __name__=='__main__':
    for X0 in [20,70,150,250,600,3000]:
        print('X0',X0)
        for E in [84,96,108,120,132]:
            print('  E%3d'%E,' '.join('H%d:%6.0f'%(H,run(X0,H,E)) for H in [300,400,500,650,800,1000]))
