import numpy as np
r=0.02; K=1700.; V=20.
def grow(X,h=12):
    return K/(1+(K/X-1)*np.exp(-r*h))
Xg=np.exp(np.linspace(np.log(5),np.log(3000),600))
fs=np.linspace(0,0.5,51)
# harvest at hours 12..132 (11), batch ends 144
Vn=np.zeros_like(Xg) # value after last harvest
pol=[]
for k in range(10,-1,-1):
    newV=np.zeros_like(Xg); bestf=np.zeros_like(Xg)
    for i,X in enumerate(Xg):
        vals=[]
        for f in fs:
            Xa=X*(1-f)
            nxt=grow(Xa) if k<10 else Xa
            v=f*X*V + (np.interp(nxt,Xg,Vn) if k<10 else 0)
            vals.append(v)
        j=int(np.argmax(vals)); newV[i]=vals[j]; bestf[i]=fs[j]
    Vn=newV; pol.insert(0,bestf)
for X0 in [20,70,150,250,600,1700]:
    X=grow(X0); tot=0; s=[]
    for k in range(11):
        f=float(np.interp(X,Xg,pol[k])); tot+=f*X*V; s.append('%d:%.2f'%(X,f)); X=grow(X*(1-f)) 
    print(X0,int(tot),' '.join(s))
print('post-harvest target per harvest index (X after harvest) for X in grid')
for k in range(11):
    row=[]
    for X in [50,100,200,300,400,600,800,1000,1500,2500]:
        f=float(np.interp(X,Xg,pol[k])); row.append('%4d/%.2f'%(X*(1-f),f))
    print('h%3d'%(12*(k+1)),' '.join(row))
