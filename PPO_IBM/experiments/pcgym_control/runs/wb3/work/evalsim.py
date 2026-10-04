"""Offline evaluation of a controller file in MY simulator (no plant budget used)."""
import sys, time, json, importlib.util
import numpy as np
import mysim


def load(path):
    spec = importlib.util.spec_from_file_location("ctl", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.Controller


def evaluate(path, seeds, params=None, perfect=False):
    C = load(path)
    res = [mysim.run(C(params), s, perfect=perfect) for s in seeds]
    c = np.array([r["cost"] for r in res])
    return c, sum(r["runaway"] for r in res), max(r["t_max"] for r in res)


if __name__ == "__main__":
    path = sys.argv[1]; n = int(sys.argv[2]); s0 = int(sys.argv[3])
    params = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    perfect = bool(params and params.get("oracle"))
    t = time.time()
    c, nr, tm = evaluate(path, range(s0, s0 + n), params, perfect)
    print("%s %s n=%d mean %.4f median %.4f max %.3f sem %.4f runaways %d tmax %.1f  (%.1fs)" % (
        path, sys.argv[4] if len(sys.argv) > 4 else "", n, c.mean(), np.median(c), c.max(), c.std() / np.sqrt(n), nr, tm, time.time() - t))
