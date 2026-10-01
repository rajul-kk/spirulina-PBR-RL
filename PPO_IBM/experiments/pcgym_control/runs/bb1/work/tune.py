import simlab, sys, numpy as np
from controller import Controller
V = {
 "A default": {},
 "B q*2": dict(q_caf=0.0018, q_tf=0.14),
 "C q*2": dict(q_caf=0.0036, q_tf=0.28),
 "D H20 fine": dict(blocks=(1, 1, 1, 1, 2, 2, 4, 8)),
 "E H8": dict(blocks=(1, 1, 2, 4)),
 "F w_du0": dict(w_du=0.0),
 "G iters6": dict(iters=6),
 "H qx big": dict(q_ca=1e-3, q_t=0.05),
 "J B+F": dict(q_caf=0.0009, q_tf=0.07, w_du=0.05),
 "K q/4+F": dict(q_caf=0.00045, q_tf=0.035, w_du=0.05),
 "L w_du.1": dict(w_du=0.1),
 "M w_du.2": dict(w_du=0.2),
 "N q/2": dict(q_caf=0.00045, q_tf=0.035),
 "P jump9 q/4": dict(q_caf=0.00045, q_tf=0.035, jump_thr=9.0),
 "Q jump9 q/8": dict(q_caf=0.00022, q_tf=0.018, jump_thr=9.0),
 "R jump6 q/8": dict(q_caf=0.00022, q_tf=0.018, jump_thr=6.0),
 "S jump12 q/8": dict(q_caf=0.00022, q_tf=0.018, jump_thr=12.0),
 "T jump9 q/8 nodu": dict(q_caf=0.00022, q_tf=0.018, jump_thr=9.0, w_du=0.0),
 "U jump9 q/20": dict(q_caf=0.0001, q_tf=0.007, jump_thr=9.0),
 "I H30": dict(blocks=(1, 1, 1, 2, 3, 6, 16)),
}
names = sys.argv[1:] or list(V)
seeds = range(100, 160)
for n in names:
    key = [k for k in V if k.startswith(n)][0]
    r = np.array([simlab.run_batch(Controller, V[key], s) for s in seeds])
    print("%-12s mean %.4f median %.4f max %.3f Tmax %.1f" % (key, r[:, 0].mean(), np.median(r[:, 0]), r[:, 0].max(), r[:, 1].max()), flush=True)
