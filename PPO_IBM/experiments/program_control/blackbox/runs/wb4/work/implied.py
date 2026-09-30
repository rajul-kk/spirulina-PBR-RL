"""Infer the controller's density estimate at interior harvest decisions:
F = 1 - xt/Xp  =>  Xp = xt/(1-F). Compare with the true pre-harvest density (harvested/(F*20))."""
import json, sys
tag = sys.argv[1]; Xt = float(sys.argv[2]); late = (0.85, 0.45)
rat = []
for l in open("../trials/results.jsonl"):
    r = json.loads(l)
    if tag not in r["batch"] or r["inoculum"] > 1000: continue
    for i, e in enumerate(r["harvests"]):
        ev = i + 1
        if ev >= 10 or e["harvested_mg"] <= 0: continue
        xt = Xt * (late[0] if ev == 8 else late[1] if ev == 9 else 1.0)
        truth = e["lab_dry_weight_mg_per_L"]
        F_real = e["harvested_mg"] / (truth * 20)
        if 0.03 < F_real < 0.45:
            Xp = xt / (1 - F_real)
            rat.append(Xp / truth)
            print(r["batch"], ev, "assay %.0f  implied est %.0f  ratio %.2f" % (truth, Xp, Xp / truth))
import statistics
print("median ratio", statistics.median(rat), "n", len(rat))
