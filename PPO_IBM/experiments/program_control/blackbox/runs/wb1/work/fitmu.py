import json, sys
from calib import replay, TR
res = {json.loads(l)["batch"]: json.loads(l) for l in open(TR + "results.jsonl")}
for b in sys.argv[1:]:
    best = min((replay(b, res[b]["inoculum"], [e["harvested_mg"] for e in res[b]["harvests"]], mumax=m, show=False), m) for m in [0.026+0.002*i for i in range(15)])
    print(b, res[b]["inoculum"], "best mumax(k=0.786)", best[1], "rmse log", round(best[0],3))
