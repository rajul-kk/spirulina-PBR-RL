import csv, glob, math, numpy as np
def est(rows, kr, gfix=None):
    c = 1.0; out = []; hist = []; g = 0.025
    for i, r in enumerate(rows):
        tu = float(r["turbidity_ntu"]); rpm = float(r["stir_rpm"])
        rr = min(tu / 250, 15); od0 = rr / max(1 - 0.05 * rr, 0.25)
        X = 300 * od0 * c ** (1 / 3)
        if float(r["harvest_frac"]) > 0 and i and float(rows[i-1]["turbidity_ntu"]) > tu * 1.3: hist = []
        hist.append(math.log(max(X, 1)));
        if len(hist) > 6: hist.pop(0)
        if len(hist) >= 4:
            n = len(hist); xs = np.arange(n); g = 0.7 * g + 0.3 * np.polyfit(xs, hist, 1)[0]
            g = min(max(g, -0.02), 0.06)
        gg = g if gfix is None else gfix
        od = X / 300
        for _ in range(50):
            stick = od * 0.05 * max(0.1, 1 - rpm / 250)
            brk = 0.5 * max(0, (rpm - 80) / 120) ** 2 * c ** 0.5 + 0.005 * max(c - 1, 0) ** 0.5
            c = max(1.0, c + (stick - brk - kr * max(gg, 0) * (c - 1)) * 0.02)
        out.append(X / (300 * float(r["true_od"])))
    return np.array(out)
files = [f for f in sorted(glob.glob("../trials/*.csv")) if "5000" not in f and "b000" not in f]
for kr in [1.0, 0.8, 0.6, 0.4]:
    E = [est(list(csv.DictReader(open(f))), kr) for f in files]
    seg = lambda a, b: np.median(np.concatenate([e[a:b] for e in E]))
    print("kr", kr, " med est/true by day:", " ".join("%.3f" % seg(24 * d, 24 * d + 24) for d in range(6)),
          " p10-p90 day4-5: %.2f-%.2f" % tuple(np.percentile(np.concatenate([e[96:] for e in E]), [10, 90])))
