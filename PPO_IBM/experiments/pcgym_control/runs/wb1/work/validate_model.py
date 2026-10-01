"""Check my plant model against privileged pilot traces: one-step prediction error."""
import glob, sys
import numpy as np
from plantsim import step_true
pat = sys.argv[1] if len(sys.argv) > 1 else "../trials/b*.csv"
eca, eT = [], []
for f in sorted(glob.glob(pat)):
    d = np.genfromtxt(f, delimiter=",", names=True)
    for k in range(len(d) - 1):
        ca, T = step_true(d["Ca_true"][k], d["T_true"][k], d["Tc"][k + 1], d["Ti"][k + 1], d["Caf"][k + 1])
        eca.append(ca - d["Ca_true"][k + 1]); eT.append(T - d["T_true"][k + 1])
    nz_ca = np.std(d["Ca"][1:] - d["Ca_true"][:-1]); nz_T = np.std(d["T"][1:] - d["T_true"][:-1])
    print(f, "noise std (obs k+1 vs truth k):", round(nz_ca, 5), round(nz_T, 4))
print("one-step max |err| Ca %.3e  T %.3e" % (np.abs(eca).max(), np.abs(eT).max()))
