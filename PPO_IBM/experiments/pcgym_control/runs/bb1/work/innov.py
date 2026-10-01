"""Model check on closed-loop data: one-step prediction errors of the fixed model (EKF replay), split into
transient samples (jacket saturated) and quiet samples; mean (bias) and lag-1 autocorrelation."""
import glob, numpy as np
from controller import Controller
I = []; SAT = []; HI = []
for f in sorted(glob.glob("../trials/b*_final.csv")):
    d = np.genfromtxt(f, delimiter=",", names=True); c = Controller(dict(r_ca=0.0021, r_t=0.21)); N = len(d)
    for k in range(N):
        if k == 0:
            c.x = np.array([d["Ca"][0], d["T"][0], c.p["caf0"], c.p["tf0"]]); c.P = np.diag([c.p["r_ca"]**2, c.p["r_t"]**2, c.p["p0_caf"]**2, c.p["p0_tf"]**2])
        else:
            c._predict(d["Tc"][k-1]); 
            if k > 10: I.append((d["Ca"][k]-c.x[0], d["T"][k]-c.x[1])); SAT.append(d["Tc"][k-1] >= 301.99 or d["Tc"][k-1] <= 295.01); HI.append(d["Tc"][k-1] >= 301.99)
            c._update(d["Ca"][k], d["T"][k])
I = np.array(I); SAT = np.array(SAT); HI = np.array(HI)
for name, m in [("quiet", ~SAT), ("sat hi (302)", SAT & HI), ("sat lo (295)", SAT & ~HI)]:
    x = I[m]; print("%-13s n %5d  Ca innov mean %+.5f std %.5f | T innov mean %+.3f std %.3f" % (name, m.sum(), x[:, 0].mean(), x[:, 0].std(), x[:, 1].mean(), x[:, 1].std()))
print("lag-1 autocorr (all): Ca %.3f T %.3f" % (np.corrcoef(I[:-1, 0], I[1:, 0])[0, 1], np.corrcoef(I[:-1, 1], I[1:, 1])[0, 1]))
