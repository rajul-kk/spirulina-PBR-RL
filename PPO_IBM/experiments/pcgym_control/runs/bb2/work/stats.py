"""Cost statistics per label from runs/bb2/trials/results.jsonl."""
import json, sys
import numpy as np
from collections import OrderedDict
g = OrderedDict()
for line in open("runs/bb2/trials/results.jsonl"):
    r = json.loads(line)
    g.setdefault(r["batch"].split("_", 1)[1], []).append(r)
for k, v in g.items():
    c = np.array([r["cost"] for r in v])
    print("%-12s n=%3d mean %.4f median %.4f sd %.4f min %.3f max %.3f  runaways %d  maxT %.1f  errors %d" % (
        k, len(c), c.mean(), np.median(c), c.std(), c.min(), c.max(), sum(r["runaway"] for r in v),
        max(r["t_max"] for r in v), sum(r["error"] is not None for r in v)))
