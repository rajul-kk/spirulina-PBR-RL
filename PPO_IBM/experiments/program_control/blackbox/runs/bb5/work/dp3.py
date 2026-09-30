import numpy as np, sys
exec(open('dp2.py').read().split("def run")[0].replace("S=float(sys.argv[1]) if len(sys.argv)>1 else 1.0","S=float(sys.argv[1])"))
for hr,best in pol:
    if hr not in (84,96,108,120): continue
    s=[]
    for x in range(50,1500,50):
        xg=grow(x); s.append('%d:%.1f'%(xg,best[np.argmin(abs(grid-x))]))
    print(hr,' '.join(s))
