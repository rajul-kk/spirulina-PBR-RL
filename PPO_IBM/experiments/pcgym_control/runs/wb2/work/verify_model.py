"""Check my plant model against the privileged trial logs: one-step-ahead prediction of the true
state from the previous true state, the applied jacket temperature and the logged disturbances."""
import csv
import glob
import sys

import numpy as np

from sim import plant_step

pat = sys.argv[1] if len(sys.argv) > 1 else "../trials/b*.csv"
eca, eT, nca, nT = [], [], [], []
for fn in sorted(glob.glob(pat)):
    rows = list(csv.DictReader(open(fn)))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    cost = np.mean(((d["Ca_true"] - d["Ca_sp"]) / 0.01) ** 2)
    for k in range(1, len(rows)):
        ca, T = plant_step(d["Ca_true"][k - 1], d["T_true"][k - 1], d["Tc"][k], d["Ti"][k], d["Caf"][k])
        eca.append(ca - d["Ca_true"][k]); eT.append(T - d["T_true"][k])
        nca.append(d["Ca"][k] - d["Ca_true"][k - 1]); nT.append(d["T"][k] - d["T_true"][k - 1])
    print(fn[-14:], "cost recomputed %.4f" % cost, "x0 guess: obs0", d["Ca"][0], d["T"][0])
eca, eT, nca, nT = map(np.array, (eca, eT, nca, nT))
print("one-step model error: Ca max|e| %.2e  T max|e| %.2e" % (np.abs(eca).max(), np.abs(eT).max()))
print("noise: Ca sd %.5f mean %.5f | T sd %.4f mean %.4f" % (nca.std(), nca.mean(), nT.std(), nT.mean()))
