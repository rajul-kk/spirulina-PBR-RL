"""Disturbance statistics from logged batches: replay EKF (default tuning), print Caf/Tf estimates every 8 samples."""
import sys, glob, numpy as np
from controller import Controller
np.set_printoptions(precision=3, suppress=True, linewidth=250)
A = []; B = []
for f in sorted(glob.glob(f"../trials/b*_{sys.argv[1]}.csv")):
    d = np.genfromtxt(f, delimiter=",", names=True)
    c = Controller(dict(q_caf=0.0018, q_tf=0.14)); est = []
    for k in range(len(d)):
        if k == 0:
            c.x = np.array([d["Ca"][0], d["T"][0], c.p["caf0"], c.p["tf0"]]); c.P = np.diag([c.p["r_ca"]**2, c.p["r_t"]**2, c.p["p0_caf"]**2, c.p["p0_tf"]**2])
        else:
            c._predict(d["Tc"][k-1]); c._update(d["Ca"][k], d["T"][k])
        est.append(c.x.copy())
    est = np.array(est); A.append(est[7::8, 2]); B.append(est[7::8, 3])
A = np.array(A); B = np.array(B)
print("Caf est (cols = sample 7,15,...,119):\n", A); print("Tf est:\n", B.round(1))
print("Caf: start(t=15) min/mean/max %.3f %.3f %.3f ; overall min/max %.3f %.3f" % (A[:, 1].min(), A[:, 1].mean(), A[:, 1].max(), A[:, 1:].min(), A[:, 1:].max()))
print("Tf : start(t=15) min/mean/max %.1f %.1f %.1f ; overall min/max %.1f %.1f" % (B[:, 1].min(), B[:, 1].mean(), B[:, 1].max(), B[:, 1:].min(), B[:, 1:].max()))
