"""Analyse plant logs (runs/wb3/trials only): scenario statistics, noise, and replay of a
controller on the logged measurements (checks determinism and estimator accuracy).
usage: analyze_trials.py <glob under ../trials> [controller.py]"""
import glob, sys
import numpy as np
import evalsim

pat = sys.argv[1]
ctl = sys.argv[2] if len(sys.argv) > 2 else None
C = evalsim.load(ctl) if ctl else None
costs, spc, dTc, dCc, nT, nC = [], [], [], [], [], []
nca, nt, uerr, eTi, eCaf, eCa, eT = [], [], [], [], [], [], []
x0 = []
for fn in sorted(glob.glob("../trials/" + pat)):
    d = np.genfromtxt(fn, delimiter=",", names=True)
    costs.append(np.mean(((d["Ca_true"] - d["Ca_sp"]) / 0.01) ** 2))
    spc += list(np.flatnonzero(np.diff(d["Ca_sp"]) != 0) + 1)
    a = list(np.flatnonzero(np.diff(d["Ti"]) != 0) + 1); dTc += a; nT.append(len(a))
    a = list(np.flatnonzero(np.diff(d["Caf"]) != 0) + 1); dCc += a; nC.append(len(a))
    nca += list(d["Ca"][1:] - d["Ca_true"][:-1]); nt += list(d["T"][1:] - d["T_true"][:-1])
    x0.append((d["Ca"][0], d["T"][0], d["Ca_sp"].min(), d["Ca_sp"].max(), d["Ti"].min(), d["Ti"].max(), d["Caf"].min(), d["Caf"].max()))
    if C:
        c = C()
        for k in range(len(d)):
            u = c.act({"t_min": d["t_min"][k], "Ca": d["Ca"][k], "T": d["T"][k], "Ca_sp": d["Ca_sp"][k]})
            uerr.append(abs(min(max(u, 295.0), 302.0) - d["Tc"][k]))
            # after act(k) the controller holds a prediction of the state at k+1
            x = c._estimate()
            eCa.append(x[0] - d["Ca_true"][k]); eT.append(x[1] - d["T_true"][k])
            if k + 1 < len(d):
                eTi.append(x[2] - d["Ti"][k + 1]); eCaf.append(x[3] - d["Caf"][k + 1])
x0 = np.array(x0)
print("batches %d  cost mean %.4f median %.4f max %.3f" % (len(costs), np.mean(costs), np.median(costs), np.max(costs)))
print("sp changes: min idx %d max idx %d; per batch %.2f" % (min(spc), max(spc), len(spc) / len(costs)))
print("Ti shifts: idx %d..%d, per batch counts %s; Caf shifts: idx %d..%d, counts %s" % (
    min(dTc), max(dTc), np.bincount(nT), min(dCc), max(dCc), np.bincount(nC)))
print("noise sd Ca %.5f (mean %.5f)  T %.4f (mean %.4f)" % (np.std(nca), np.mean(nca), np.std(nt), np.mean(nt)))
print("ranges: Ca0 %.3f-%.3f T0 %.1f-%.1f sp %.3f-%.3f Ti %.2f-%.2f Caf %.3f-%.3f" % (
    x0[:, 0].min(), x0[:, 0].max(), x0[:, 1].min(), x0[:, 1].max(), x0[:, 2].min(), x0[:, 3].max(),
    x0[:, 4].min(), x0[:, 5].max(), x0[:, 6].min(), x0[:, 7].max()))
if C:
    print("replay of %s on logged measurements: max |Tc - logged| = %.2e" % (ctl, max(uerr)))
    print("one-step-ahead prediction rms: Ca %.5f  T %.4f | disturbance estimate rms: Ti %.3f  Caf %.5f" % (
        np.sqrt(np.mean(np.square(eCa))), np.sqrt(np.mean(np.square(eT))),
        np.sqrt(np.mean(np.square(eTi))), np.sqrt(np.mean(np.square(eCaf)))))
