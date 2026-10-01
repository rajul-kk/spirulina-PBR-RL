import sys, json, numpy as np
r = [json.loads(l) for l in open("../trials/results.jsonl")]
for lab in sys.argv[1:]:
    c = np.array([x["cost"] for x in r if x["batch"].endswith("_"+lab)]); t = np.array([x["t_max"] for x in r if x["batch"].endswith("_"+lab)])
    ra = sum(bool(x["runaway"]) for x in r if x["batch"].endswith("_"+lab)); er = sum(x["error"] is not None for x in r if x["batch"].endswith("_"+lab))
    print(f"{lab}: n {len(c)} mean {c.mean():.4f} (SE {c.std(ddof=1)/len(c)**.5:.4f}) median {np.median(c):.4f} p90 {np.percentile(c,90):.3f} max {c.max():.3f} min {c.min():.3f} | Tmax {t.max():.1f} runaways {ra} errors {er}")
