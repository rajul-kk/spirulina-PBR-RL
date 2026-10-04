"""Replay privileged trial logs through my own plant model: one-step prediction error."""
import glob, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("check_model.py", 1)[0])
from sim import plant_step
import controller as C
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
for f in sorted(glob.glob(__file__.rsplit("work", 1)[0] + "trials/" + pat + ".csv")):
    d = np.genfromtxt(f, delimiter=",", names=True)
    e_ca, e_T, e4c, e4t = [], [], [], []
    for k in range(len(d) - 1):
        # row k: obs at k, Tc applied, truth = state after the step with Ti[k], Caf[k]
        ca, T = d["Ca_true"][k], d["T_true"][k]
        cn, Tn = plant_step(ca, T, d["Tc"][k + 1], d["Ti"][k + 1], d["Caf"][k + 1])
        e_ca.append(cn - d["Ca_true"][k + 1]); e_T.append(Tn - d["T_true"][k + 1])
        cn, Tn = C._step(ca, T, d["Tc"][k + 1], d["Ti"][k + 1], d["Caf"][k + 1], 4)
        e4c.append(cn - d["Ca_true"][k + 1]); e4t.append(Tn - d["T_true"][k + 1])
    nz_ca = np.std(d["Ca"][1:] - d["Ca_true"][:-1]); nz_T = np.std(d["T"][1:] - d["T_true"][:-1])
    cost = np.mean(((d["Ca_true"] - d["Ca_sp"]) / 0.01) ** 2)
    print(f.split("trials")[-1][1:], "max|err| Ca %.2e T %.2e | nsub4 Ca %.2e T %.2e | noise sd Ca %.4f T %.3f | cost %.3f"
          % (np.abs(e_ca).max(), np.abs(e_T).max(), np.abs(e4c).max(), np.abs(e4t).max(), nz_ca, nz_T, cost))
