import numpy as np, sys
px=[0,50,100,150,250,400,600,1000,1300,1800,2300,2900,4000]
py=[0,1.1,2.3,3.5,5.5,6.3,7.0,7.2,7.0,5.0,2.5,0,-1]
S=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
def P(x): return S*np.interp(x,px,py)
def grow(x,h=12.0):
    for _ in range(24): x=x+0.5*P(x)
    return x
grid=np.linspace(0,3500,351)
fr=np.linspace(0,0.5,11)
V=np.zeros_like(grid); pol=[]
for k in range(11,0,-1):
    newV=np.zeros_like(grid); best=np.zeros_like(grid)
    for i,x in enumerate(grid):
        xg=grow(x)
        vals=[f*xg*20 + np.interp(xg*(1-f),grid,V) for f in fr]
        j=int(np.argmax(vals)); newV[i]=vals[j]; best[i]=fr[j]
    V=newV; pol.append((12*k,best))
pol=pol[::-1]
def run(X0, rule):
    x=X0; tot=0; s=[]
    for (hr,best) in pol:
        xg=grow(x); f=rule(hr,xg,x,best)
        tot+=f*xg*20; s.append('%.0f/%.1f'%(xg,f)); x=xg*(1-f)
    return tot,s
opt=lambda hr,xg,x,best: best[np.argmin(abs(grid-x))]
A=lambda hr,xg,x,best: 0.5 if hr>=108 else 0.0
B=lambda hr,xg,x,best: 0.5 if hr>=96 else 0.0
C=lambda hr,xg,x,best: 0.5 if (hr>=120 or (hr==108 and xg>350) or (hr==96 and xg>700)) else 0.0
for X0 in [20,60,130,250,600,1000,1800,2900]:
    to,s=run(X0,opt)
    print(X0,'opt %.0f A %.0f B %.0f C %.0f'%(to,run(X0,A)[0],run(X0,B)[0],run(X0,C)[0]),' '.join(s))
