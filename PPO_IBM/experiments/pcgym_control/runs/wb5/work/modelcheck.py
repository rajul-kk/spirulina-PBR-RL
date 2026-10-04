import glob, numpy as np, csv
from mysim import step
for f in sorted(glob.glob('../trials/b*.csv')):
    rows=list(csv.DictReader(open(f)))
    err=[]
    for a,b in zip(rows[:-1],rows[1:]):
        x=np.array([float(a['Ca_true']),float(a['T_true'])])
        xp=step(x,float(b['Tc']),float(b['Ti']),float(b['Caf']),nsub=8)
        err.append([xp[0]-float(b['Ca_true']),xp[1]-float(b['T_true'])])
    err=np.array(err)
    nCa=np.std([float(r['Ca'])-float(r['Ca_true']) for r in rows[1:]])
    print(f, "max|dCa| %.2e max|dT| %.2e"%tuple(np.abs(err).max(0)))
