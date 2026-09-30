from mfsim import run
from ana import load
import sys
# replicate probe: light ramp by od_est, stir fixed, harvest 0.5 from 108h
def mk(stir):
    st = {'L': 400.0}
    def pol(t, od, c):
        est = od * c**(-1/3)
        pts=[(0.1,600),(0.25,800),(0.5,1100),(0.75,1400),(1.0,1700)]
        import numpy as np
        tgt = float(np.interp(est,[p[0] for p in pts],[p[1] for p in pts]))
        st['L'] = min(tgt, st['L']+100*0.1) if tgt>st['L'] else tgt
        return stir, st['L'], (0.5 if t >= 108 else 0.0)
    return pol
for stir, b in ((90,'b001_p90'),(60,'b008_p60'),(120,'b006_p120')):
    tot, od, tr = run(mk(stir), 0.415, mu_eff=float(sys.argv[1]) if len(sys.argv)>1 else 0.031, trace=True)
    r = {int(x['hour']): x for x in load(b)}
    print(stir, round(tot), ' '.join(f"{t:.0f}:{o:.2f}({r[int(t)]['true_od']:.2f})" for t,o,c,m,T in tr[::2] if int(t) in r))
