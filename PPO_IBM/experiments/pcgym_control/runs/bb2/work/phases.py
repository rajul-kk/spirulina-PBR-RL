"""Split logged (measured) squared error by phase: start-up, after setpoint changes, rest.
Contribution is in cost units (sum over phase / 120). Analyser noise adds ~0.05-0.07 in total."""
import sys, glob, json
import numpy as np
sys.path.insert(0, "runs/bb2/work")
from show import load

W = 10
costs = {}
for line in open("runs/bb2/trials/results.jsonl"):
    r = json.loads(line)
    costs[r["batch"]] = r["cost"]
tot = np.zeros(4)
nb = 0
for pat in sys.argv[1:]:
    for p in sorted(glob.glob("runs/bb2/trials/" + pat)):
        d = load(p)
        sp = d["Ca_sp"]
        e2 = ((d["Ca"] - sp) / 0.01) ** 2
        ch = [i for i in range(1, 120) if sp[i] != sp[i - 1]]
        cat = np.zeros(120, int)
        cat[:W] = 1
        for c in ch:
            cat[c:c + W] = 2
        sat = (d["Tc"] <= 295.01) | (d["Tc"] >= 301.99)
        parts = [e2[cat == k].sum() / 120 for k in (1, 2, 0)]
        rest_sat = e2[(cat == 0) & sat].sum() / 120
        name = p.replace("\\", "/").split("/")[-1][:-4]
        print("%-16s true %.3f meas %.3f | start %.3f spchg %.3f rest %.3f (of which saturated %.3f, n_sat %d) | steps %s" % (
            name, costs.get(name, float("nan")), e2.mean(), parts[0], parts[1], parts[2], rest_sat,
            int(((cat == 0) & sat).sum()), " ".join("%+.3f" % (sp[c] - sp[c - 1]) for c in ch)))
        tot += np.array(parts + [rest_sat])
        nb += 1
print("MEAN start %.3f spchg %.3f rest %.3f (rest saturated %.3f)  over %d batches" % (tuple(tot / nb) + (nb,)))
