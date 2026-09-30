import numpy as np
px=[0,50,100,150,250,375,525,800,1300,1600,2000,2500,3000,4000]
py=[0,1.1,2.3,3.4,5.2,6.5,8.0,8.5,9.5,6.0,4.0,2.0,0,-1]
def P(x): return np.interp(x,px,py)
def grow(x,h=12.0):
    for _ in range(int(h/0.5)): x=x+0.5*P(x)
    return x
grid=np.linspace(0,3500,701)
fr=np.linspace(0,0.5,11)
V=np.zeros_like(grid)   # value after last harvest at 132 (nothing)
pol=[]
for k in range(11,0,-1):   # harvest k at hour 12k, before it grow 12h
    newV=np.zeros_like(grid); best=np.zeros_like(grid)
    for i,x in enumerate(grid):
        xg=grow(x)
        vals=[f*xg*20 + np.interp(xg*(1-f),grid,V) for f in fr]
        j=int(np.argmax(vals)); newV[i]=vals[j]; best[i]=fr[j]
    V=newV; pol.append((12*k,best))
pol=pol[::-1]
for X0 in [20,60,130,250,600,1000,1800,2900]:
    x=X0; tot=0; s=[]
    for (hr,best) in pol:
        xg=grow(x); f=float(np.interp(x,grid,best)); f=best[np.argmin(abs(grid-x))]
        tot+=f*xg*20; s.append('%d:%.0f/%.1f'%(hr,xg,f)); x=xg*(1-f)
    print(X0, 'tot %.0f'%tot, ' '.join(s))
print('policy: hour -> frac at pre-growth X (state after previous harvest)')
for hr,best in pol:
    s=[]
    for X in [50,100,200,300,400,500,600,800,1000,1200,1400,1700,2000,2500]:
        s.append('%d:%.1f'%(X,best[np.argmin(abs(grid-X))]))
    print(hr,' '.join(s))
