"""Summarise pilot-plant results from runs/wb2/trials/results.jsonl by label.
usage: python summary.py [label-substring]"""
import json
import sys

import numpy as np

key = sys.argv[1] if len(sys.argv) > 1 else ""
rows = [json.loads(l) for l in open("../trials/results.jsonl") if l.strip()]
rows = [r for r in rows if key in r["batch"]]
c = np.array([r["cost"] for r in rows])
print("label '%s': n %d  mean %.4f (se %.4f)  median %.4f  p90 %.4f  max %.4f  min %.4f" % (
    key, len(c), c.mean(), c.std(ddof=1) / np.sqrt(len(c)), np.median(c), np.percentile(c, 90), c.max(), c.min()))
print("runaways %d  highest T %.2f K  controller errors %d" % (
    sum(r["runaway"] for r in rows), max(r["t_max"] for r in rows), sum(r["error"] is not None for r in rows)))
