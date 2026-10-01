import simlab, sys, numpy as np
from controller import Controller
R = dict(r_ca=0.0021, r_t=0.21)
V = {
 "a old R": {},
 "b new R": R,
 "c R q/2": dict(R, q_caf=0.00045, q_tf=0.035),
 "d R q*2": dict(R, q_caf=0.0018, q_tf=0.14),
 "e R w_du.1": dict(R, w_du=0.1),
 "f R w_du0": dict(R, w_du=0.0),
 "g R jump9 q/4": dict(R, q_caf=0.00022, q_tf=0.018, jump_thr=9.0),
 "h R jump7 q/4 big": dict(R, q_caf=0.00022, q_tf=0.018, jump_thr=7.0, jump_caf=0.02, jump_tf=1.5),
 "k ML": dict(r_ca=0.002, r_t=0.2, q_ca=2.4e-4, q_t=0.0084, q_caf=0.00195, q_tf=0.114),
 "l ML w_du0": dict(r_ca=0.002, r_t=0.2, q_ca=2.4e-4, q_t=0.0084, q_caf=0.00195, q_tf=0.114, w_du=0.0),
 "i R qtf*2": dict(R, q_tf=0.14),
 "j R qcaf*2": dict(R, q_caf=0.0018),
}
names = sys.argv[1:] or list(V)
seeds = range(500, 600); base = None
for n in names:
    key = [k for k in V if k.startswith(n)][0]
    r = np.array([simlab.run_batch(Controller, V[key], s)[0] for s in seeds])
    if base is None: base = r
    d = r - base
    print("%-18s mean %.4f median %.4f max %.3f | paired diff vs first %+.4f (SE %.4f)" % (key, r.mean(), np.median(r), r.max(), d.mean(), d.std(ddof=1)/len(d)**.5), flush=True)
