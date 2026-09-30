"""Empirical productivity P(X) = dX/dt (mg/L/h) from privileged true OD, excluding harvest hours."""
import csv, glob, numpy as np, sys
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
bins = [0, 50, 100, 150, 225, 300, 400, 500, 600, 700, 800, 1000, 1300, 2000, 4000]
acc = {i: [] for i in range(len(bins) - 1)}
for f in glob.glob(f"../trials/{pat}.csv"):
    rows = list(csv.DictReader(open(f)))
    X = np.array([300 * float(r["true_od"]) for r in rows]); h = np.array([float(r["hour"]) for r in rows])
    for i in range(len(X) - 3):
        # skip windows spanning a harvest event (multiples of 12 h)
        if int(h[i] // 12) != int(h[i + 3] // 12) or h[i] < 1: continue
        P = (X[i + 3] - X[i]) / (h[i + 3] - h[i]); xm = 0.5 * (X[i] + X[i + 3])
        b = np.searchsorted(bins, xm) - 1
        if 0 <= b < len(bins) - 1: acc[b].append((P, float(rows[i]["stir_rpm"]), float(rows[i]["light_umol"])))
for b in acc:
    if acc[b]:
        a = np.array(acc[b])
        print("X %4d-%4d n %4d  P median %5.2f  (p25 %5.2f p75 %5.2f)  mu %.4f  rpm %.0f light %.0f" % (
            bins[b], bins[b + 1], len(a), np.median(a[:, 0]), np.percentile(a[:, 0], 25), np.percentile(a[:, 0], 75),
            np.median(a[:, 0]) / (0.5 * (bins[b] + bins[b + 1])), np.median(a[:, 1]), np.median(a[:, 2])))
