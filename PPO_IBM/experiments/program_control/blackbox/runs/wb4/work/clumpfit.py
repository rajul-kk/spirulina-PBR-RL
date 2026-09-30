import csv, glob, math, sys, numpy as np
def replay(rows, use_cells=True, gmode="cells"):
    c = 1.0; out = []; prev_cells = None; prev_od = None
    for r in rows:
        od = float(r["true_od"]); rpm = float(r["stir_rpm"]); cells = int(r["true_cells"])
        if prev_cells is not None:
            # divisions: births reset clump mean
            if gmode == "cells" and cells > prev_cells:
                born = cells - prev_cells
                c = (c * prev_cells + born) / cells
            elif gmode == "mass" and od > prev_od and float(r["harvest_frac"]) >= 0:
                g = math.log(od / prev_od)
                c = 1 + (c - 1) * math.exp(-max(g, 0))
            for _ in range(50):
                stick = od * 0.05 * max(0.1, 1 - rpm / 250)
                brk = 0.5 * max(0, (rpm - 80) / 120) ** 2 * c ** 0.5 + 0.005 * max(c - 1, 0) ** 0.5
                c = max(1.0, c + (stick - brk) * 0.02)
        prev_cells, prev_od = cells, od
        tu = float(r["turbidity_ntu"])
        pred = od * c ** (-1 / 3) / (1 + 0.05 * od) * 250
        out.append((float(r["hour"]), tu / max(pred, 1e-9), c))
    return out
for f in sorted(glob.glob("../trials/*.csv")):
    rows = list(csv.DictReader(open(f)))
    for mode in ["cells", "mass"]:
        o = replay(rows, gmode=mode)
        rat = np.array([x[1] for x in o])
        print(f.split("/")[-1][-12:], mode, "ratio mean %.3f sd %.3f  first %.2f mid %.2f end %.2f  cmax %.1f" % (
            rat.mean(), rat.std(), rat[:10].mean(), rat[60:80].mean(), rat[-10:].mean(), max(x[2] for x in o)))
