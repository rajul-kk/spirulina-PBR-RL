"""Strain-adjusted comparison: early specific growth rate (true OD, 3-30 h, before any harvest and
before density limits; the light policy is identical across arms there) as covariate."""
import json, csv, sys, numpy as np
res = [json.loads(l) for l in open("../trials/results.jsonl")]
pat = sys.argv[1] if len(sys.argv) > 1 else "_ab_"
rows = []
for r in res:
    if pat not in r["batch"]: continue
    R = list(csv.DictReader(open(f"../trials/{r['batch']}.csv")))
    od = {float(x["hour"]): float(x["true_od"]) for x in R}
    g = np.log(od[30.0] / od[3.0]) / 27.0
    arm = r["batch"].split(pat)[1]
    rows.append((arm, g, r["total_harvested_mg"]))
arms = sorted(set(a for a, _, _ in rows))
G = np.array([x[1] for x in rows]); H = np.log([x[2] for x in rows])
A = np.array([[1.0 if x[0] == a else 0.0 for a in arms] for x in rows])
Xd = np.column_stack([A, G - G.mean()])
beta, *_ = np.linalg.lstsq(Xd, H, rcond=None)
resid = H - Xd @ beta
s2 = resid @ resid / (len(H) - Xd.shape[1])
cov = s2 * np.linalg.inv(Xd.T @ Xd)
print(f"slope dlogH/dg = {beta[-1]:.1f}, residual sd = {np.sqrt(s2):.3f} (log units)")
for i, a in enumerate(arms):
    gs = [x[1] for x in rows if x[0] == a]
    d = beta[i] - beta[0]; se = np.sqrt(cov[i, i] + cov[0, 0] - 2 * cov[i, 0])
    print(f"{a:14s} n={len(gs)} mean g {np.mean(gs):.4f} adj mean {np.exp(beta[i]):8.0f} mg  vs {arms[0]}: {d*100:+.1f}% +- {se*100:.1f}")
