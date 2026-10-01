"""For the frozen controller's pilot batches: how much of the cost is incurred while the jacket is at a limit."""
import glob, json
import numpy as np
tot = sat = 0.0
worst = (0, None)
for f in sorted(glob.glob("../trials/b*_v3.csv") + glob.glob("../trials/b*_final.csv")):
    d = np.genfromtxt(f, delimiter=",", names=True)
    sq = ((d["Ca_true"] - d["Ca_sp"]) / 0.01) ** 2
    atlim = (d["Tc"] <= 295.001) | (d["Tc"] >= 301.999)
    tot += sq.sum(); sat += sq[atlim].sum()
    if sq.mean() > worst[0]:
        worst = (sq.mean(), f, sq[atlim].sum() / sq.sum(), d["Ca_sp"][[0, 119]], np.unique(d["Ca_sp"]))
print("share of total cost incurred at a jacket limit: %.3f" % (sat / tot))
print("worst batch:", worst)
