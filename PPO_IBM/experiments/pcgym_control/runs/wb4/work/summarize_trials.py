"""Summarise pilot results for a label from runs/wb4/trials/results.jsonl."""
import json, sys
import numpy as np
lab = sys.argv[1]
rows = [json.loads(l) for l in open(__file__.rsplit("work", 1)[0] + "trials/results.jsonl")]
rows = [r for r in rows if r["batch"].endswith("_" + lab)]
c = np.array([r["cost"] for r in rows])
print("n %d mean %.4f +/- %.4f median %.4f p90 %.3f max %.3f runaways %d max T %.1f errors %d" % (
    len(c), c.mean(), c.std(ddof=1) / np.sqrt(len(c)), np.median(c), np.percentile(c, 90), c.max(),
    sum(r["runaway"] for r in rows), max(r["t_max"] for r in rows), sum(r["error"] is not None for r in rows)))
