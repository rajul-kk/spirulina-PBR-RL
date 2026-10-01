import sys, glob, numpy as np
from controller import Controller
f = glob.glob(f"../trials/{sys.argv[1]}_*.csv")[0]; a, b = int(sys.argv[2]), int(sys.argv[3])
d = np.genfromtxt(f, delimiter=",", names=True); c = Controller()
for k in range(len(d)):
    if k == 0:
        c.x = np.array([d["Ca"][0], d["T"][0], c.p["caf0"], c.p["tf0"]]); c.P = np.diag([c.p["r_ca"]**2, c.p["r_t"]**2, c.p["p0_caf"]**2, c.p["p0_tf"]**2])
    else:
        c._predict(d["Tc"][k-1]); c._update(d["Ca"][k], d["T"][k])
    if a <= k < b: print("%3d Ca %.4f (est %.4f) sp %.4f err*100 %+5.2f | T %.2f Tc %.2f | Caf^ %.4f Tf^ %.2f" % (k, d["Ca"][k], c.x[0], d["Ca_sp"][k], (c.x[0]-d["Ca_sp"][k])*100, d["T"][k], d["Tc"][k], c.x[2], c.x[3]))
