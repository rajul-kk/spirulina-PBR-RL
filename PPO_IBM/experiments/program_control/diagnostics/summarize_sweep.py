"""Table of median harvest (g, = median_mg/1000) and crash_rate per arm x reset from reset_sweep/*.json.
usage: python summarize_sweep.py [dir]"""
import glob
import json
import os
import re
import statistics
import sys

RESETS = ["30", "60", "150", "600", "never"]
PAT = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)_r(?P<r>\w+)\.json$")


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        s = json.load(open(f, encoding="utf-8"))["summary"]
        rows[(m["arm"], int(m["seed"]), m["r"])] = (s["median_mg"] / 1000.0, s["crash_rate"])
    return rows


def render(rows):
    arms = sorted({k[0] for k in rows})
    out = []
    out.append("Seed medians (harvest g / crash_rate), cell = median over seeds [n seeds]")
    out.append("%-10s" % "arm" + "".join("%22s" % ("reset " + r) for r in RESETS))
    for a in arms:
        line = "%-10s" % a
        for r in RESETS:
            v = [rows[k] for k in rows if k[0] == a and k[2] == r]
            line += "%22s" % ("%.2f / %.3f [%d]" % (statistics.median(x[0] for x in v),
                                                    statistics.median(x[1] for x in v), len(v)) if v else "-")
        out.append(line)
    out.append("")
    out.append("Per seed (harvest g / crash_rate)")
    out.append("%-10s%-6s" % ("arm", "seed") + "".join("%20s" % ("reset " + r) for r in RESETS))
    for a in arms:
        for s in sorted({k[1] for k in rows if k[0] == a}):
            line = "%-10s%-6d" % (a, s)
            for r in RESETS:
                v = rows.get((a, s, r))
                line += "%20s" % ("%.2f / %.3f" % v if v else "-")
            out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "reset_sweep")
    print(render(load(d)))
