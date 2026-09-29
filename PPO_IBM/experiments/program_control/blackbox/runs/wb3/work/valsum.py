import json, sys, numpy as np
lab = sys.argv[1:]
R = [json.loads(l) for l in open("../trials/results.jsonl")]
R = [r for r in R if any(r["batch"].endswith("_" + x) for x in lab)]
def s(rs, name):
    t = np.array([r["total_harvested_mg"] for r in rs]); lost = sum(r["culture_lost"] for r in rs); err = sum(bool(r["error"]) for r in rs)
    print(f"{name:14s} n {len(rs):3d} median {np.median(t)/1000:6.1f} g  p25 {np.percentile(t,25)/1000:6.1f} g  min {t.min()/1000:5.1f}  mean {t.mean()/1000:5.1f}  lost {lost} errors {err}")
s(R, "all")
for lo, hi, nm in [(0, 99, "30-99"), (100, 400, "100-400"), (401, 999, "401-999"), (1000, 99999, ">=1000")]:
    rs = [r for r in R if lo <= r["inoculum"] <= hi]
    if rs: s(rs, "inoc " + nm)
