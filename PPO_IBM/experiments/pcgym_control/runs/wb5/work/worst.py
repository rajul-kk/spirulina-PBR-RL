import csv, sys, numpy as np
for b in sys.argv[1:]:
    rows=list(csv.DictReader(open(f'../trials/{b}.csv')))
    a=np.array([[float(r[k]) for k in ['step','Ca_sp','Tc','Ca_true','T_true','Ti','Caf']] for r in rows])
    e=((a[:,3]-a[:,1])/.01)**2
    print(b, "cost %.3f"%e.mean(), "cost by 10-step blocks", np.round([e[i:i+10].sum()/120 for i in range(0,120,10)],3))
    for k in range(0,120,6): print("  ",np.round(a[k],3), round(e[k],2))
