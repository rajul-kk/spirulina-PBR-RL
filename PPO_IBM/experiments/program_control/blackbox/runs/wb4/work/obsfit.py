import csv, glob, math, sys, numpy as np
def est(rows):
    c = 1.0; prevX = None; out = []
    for r in rows:
        tu = float(r["turbidity_ntu"]); rpm = float(r["stir_rpm"])
        rr = min(tu / 250, 15); od0 = rr / max(1 - 0.05 * rr, 0.25)
        X = 300 * od0 * c ** (1 / 3)
        if prevX and X > prevX and float(r["harvest_frac"]) == 0 or (prevX and X > prevX):
            g = math.log(X / prevX)
            if g < 0.1: c = 1 + (c - 1) * math.exp(-g)
        prevX = X
        od = X / 300
        for _ in range(50):
            stick = od * 0.05 * max(0.1, 1 - rpm / 250)
            brk = 0.5 * max(0, (rpm - 80) / 120) ** 2 * c ** 0.5 + 0.005 * max(c - 1, 0) ** 0.5
            c = max(1.0, c + (stick - brk) * 0.02)
        out.append(X / (300 * float(r["true_od"])))
    return np.array(out)
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "../trials/*.csv")):
    e = est(list(csv.DictReader(open(f))))
    print(f[-14:], "est/true: first %.2f  24h %.2f  48h %.2f  72h %.2f 96h %.2f 120h %.2f end %.2f" % tuple(
        [e[0]] + [e[i] for i in (24, 48, 72, 96, 120)] + [e[-1]]))
