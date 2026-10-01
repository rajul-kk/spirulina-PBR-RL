"""Summarise pilot results by label from runs/wb1/trials/results.jsonl."""
import json, sys, collections
import numpy as np
rows = [json.loads(l) for l in open("../trials/results.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    by[r["batch"].split("_", 1)[1]].append(r)
for lab, rs in by.items():
    c = np.array([r["cost"] for r in rs])
    print(f"{lab}: n={len(c)} mean={c.mean():.4f} sem={c.std(ddof=1)/np.sqrt(len(c)):.4f} median={np.median(c):.4f} "
          f"p90={np.percentile(c,90):.3f} max={c.max():.3f} runaways={sum(bool(r.get('runaway')) for r in rs)} "
          f"tmax={max(r.get('t_max', 0) for r in rs):.2f} errors={sum(1 for r in rs if r.get('error'))}")
