import json, sys, numpy as np
rs = [json.loads(l) for l in open("../trials/results.jsonl")]
sel = [r for r in rs if r["controller"] == "controller.py"]
def rep(name, rr):
    h = np.array([r["total_harvested_mg"] for r in rr])
    lost = sum(r["culture_lost"] for r in rr)
    print("%-28s n=%3d  median %6.0f  p25 %6.0f  mean %6.0f  min %6.0f  lost %d" % (name, len(h), np.median(h), np.percentile(h, 25), h.mean(), h.min(), lost))
rep("natural mix (val)", [r for r in sel if r["batch"].endswith("_val")])
rep("all controller.py", sel)
for lo, hi, nm in [(30, 80, "30-80"), (81, 99, "81-99"), (100, 400, "100-400"), (401, 599, "401-599"), (600, 5000, "600-5000")]:
    rr = [r for r in sel if lo <= r["inoculum"] <= hi]
    if rr: rep("inoculum " + nm, rr)
for x in (30, 80, 250, 1000, 5000):
    rr = [r for r in sel if r["inoculum"] == x]
    if rr: rep("fixed %d" % x, rr)
