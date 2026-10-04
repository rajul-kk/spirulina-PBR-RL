"""Replay logged Tc/Ti/Caf through my model and compare with logged true states."""
import glob, sys
import numpy as np
from mysim import step
pat = sys.argv[1] if len(sys.argv) > 1 else "../trials/b*_probe.csv"
for fn in sorted(glob.glob(pat)):
    d = np.genfromtxt(fn, delimiter=",", names=True)
    e1 = e2 = 0.0
    # one-step prediction from the logged true state
    for k in range(len(d) - 1):
        ca, T = step(d["Ca_true"][k], d["T_true"][k], d["Tc"][k + 1], d["Ti"][k + 1], d["Caf"][k + 1])
        e1 = max(e1, abs(ca - d["Ca_true"][k + 1])); e2 = max(e2, abs(T - d["T_true"][k + 1]))
    # open-loop replay of whole batch from state after step 0
    ca, T = d["Ca_true"][0], d["T_true"][0]; o1 = o2 = 0.0
    for k in range(len(d) - 1):
        ca, T = step(ca, T, d["Tc"][k + 1], d["Ti"][k + 1], d["Caf"][k + 1])
        o1 = max(o1, abs(ca - d["Ca_true"][k + 1])); o2 = max(o2, abs(T - d["T_true"][k + 1]))
    print(fn[-18:], "1-step max err Ca %.2e T %.2e | open-loop Ca %.2e T %.2e" % (e1, e2, o1, o2),
          "noise sd Ca %.4f T %.3f" % (np.std(d["Ca"][1:] - d["Ca_true"][:-1]), np.std(d["T"][1:] - d["T_true"][:-1])))
