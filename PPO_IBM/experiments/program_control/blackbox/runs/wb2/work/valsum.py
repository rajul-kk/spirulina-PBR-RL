import json, sys, numpy as np
pats = sys.argv[1:] or ["_val"]
res = [json.loads(l) for l in open("../trials/results.jsonl")]
rs = [r for r in res if any(r["batch"].endswith(p) or p in r["batch"] for p in pats)]
def summ(name, sub):
    if not sub: return
    h = np.array([r["total_harvested_mg"] for r in sub]); lost = sum(r["culture_lost"] for r in sub)
    err = sum(1 for r in sub if r["error"])
    print(f"{name:22s} n={len(sub):3d} median {np.median(h):8.0f}  p25 {np.percentile(h,25):8.0f}  min {h.min():8.0f}  mean {h.mean():8.0f}  lost {lost} errors {err}")
summ("all", rs)
for lo, hi in [(30, 81), (81, 100), (100, 200), (200, 401), (401, 1000), (1000, 2500), (2500, 6000)]:
    summ(f"inoc {lo}-{hi-1}", [r for r in rs if lo <= r["inoculum"] < hi])
