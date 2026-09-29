import json, sys, numpy as np
lab = sys.argv[1:]
rs = [json.loads(l) for l in open("../trials/results.jsonl")]
rs = [r for r in rs if any(r["batch"].endswith("_" + x) for x in lab)]
def show(name, sel):
    if not sel: return
    h = np.array([r["total_harvested_mg"] for r in sel]); lost = sum(r["culture_lost"] for r in sel)
    print(f"{name:16s} n={len(sel):3d} median={np.median(h):8.0f} p25={np.percentile(h,25):8.0f} min={h.min():8.0f} lost={lost}")
show("all", rs)
for lo, hi in [(30, 80), (80, 100), (100, 400), (400, 600), (600, 2000), (2000, 5001)]:
    show(f"inoc {lo}-{hi-1}", [r for r in rs if lo <= r["inoculum"] < hi])
