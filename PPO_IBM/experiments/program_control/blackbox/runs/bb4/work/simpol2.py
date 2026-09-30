import numpy as np
from simpol import run
E={9:.5,10:.5,11:.5}
for a,b,c in [(800,3,2000),(800,3,1600),(1000,2,2000),(600,4,2000),(800,0,800),(1000,0,1000),(1200,0,1200),(900,2,1800),(700,5,2000)]:
    Lf=lambda X,a=a,b=b,c=c: min(c,a+b*X)
    res=[run(i,500,Lf,E) for i in [30,100,200,300,1000]]
    print(a,b,c,' '.join('%6.0f'%r for r in res))
for E2 in [{9:.3,10:.5,11:.5},{8:.25,9:.4,10:.5,11:.5},{9:.4,10:.45,11:.5}]:
    res=[run(i,500,lambda X: min(2000,800+3*X),E2) for i in [30,100,200,300,1000]]
    print(E2,' '.join('%6.0f'%r for r in res))
