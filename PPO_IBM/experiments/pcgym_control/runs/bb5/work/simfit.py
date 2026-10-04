import numpy as np, csv, glob, sys
def load(f):
    r=list(csv.DictReader(open(f)))
    return {k:np.array([float(x[k]) for x in r]) for k in r[0]}
P=dict(qV=1.0,k0=7.2e10,ER=8750.,J=5e4/(1000*0.239),U=5e4/(100*1000*0.239),Caf=1.0,Tf=350.)
def f(x,Tc,p):
    Ca,T=x; k=p['k0']*np.exp(-p['ER']/T)
    return np.array([p['qV']*(p['Caf']-Ca)-k*Ca, p['qV']*(p['Tf']-T)+p['J']*k*Ca+p['U']*(Tc-T)])
def step(x,Tc,p,dt=13/60,n=20):
    h=dt/n
    for _ in range(n):
        k1=f(x,Tc,p);k2=f(x+h/2*k1,Tc,p);k3=f(x+h/2*k2,Tc,p);k4=f(x+h*k3,Tc,p)
        x=x+h/6*(k1+2*k2+2*k3+k4)
    return x
if __name__=='__main__':
  for fn in sorted(glob.glob('../trials/b00*.csv')):
    d=load(fn); x=np.array([d['Ca'][0],d['T'][0]]); e=[]
    for i in range(len(d['Tc'])-1):
        # one-step-ahead prediction from measured state
        xm=np.array([d['Ca'][i],d['T'][i]]); xp=step(xm,d['Tc'][i],P)
        e.append([d['Ca'][i+1]-xp[0],d['T'][i+1]-xp[1]])
    e=np.array(e); print(fn, '1-step resid mean',e.mean(0).round(4),'std',e.std(0).round(4))
    # free run
    x=np.array([d['Ca'][0],d['T'][0]]); out=[]
    for i in range(len(d['Tc'])-1):
        x=step(x,d['Tc'][i],P); out.append(x)
    out=np.array(out); r=np.c_[d['Ca'][1:],d['T'][1:]]-out
    print('   free-run resid by quarter Ca:',[round(r[j:j+30,0].mean(),4) for j in range(0,119,30)],'T:',[round(r[j:j+30,1].mean(),2) for j in range(0,119,30)])
