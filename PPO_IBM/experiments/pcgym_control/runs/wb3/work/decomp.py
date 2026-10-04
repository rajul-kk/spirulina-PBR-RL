"""Offline: decompose cost by cause (my simulator). usage: decomp.py ctl.py n seed0 [json params]"""
import sys, json
import numpy as np
from multiprocessing import Pool
import mysim, evalsim


def one(a):
    path, seed, params = a
    C = evalsim.load(path)
    sc = mysim.scenario(seed)
    out, tr = mysim.run(C(params), seed, perfect=bool(params and params.get("oracle")), trace=True)
    sq = ((tr[:, 5] - tr[:, 3]) / 0.01) ** 2 / mysim.N
    lab = np.full(mysim.N, 3)                      # 3 = quiet
    for name, code in (("Ti", 2), ("Caf", 2), ("sp", 1)):
        for c in np.flatnonzero(np.diff(sc[name]) != 0) + 1:
            lab[c:c + 15] = code
    lab[:15] = np.where(lab[:15] == 3, 0, lab[:15])
    return out["cost"], [sq[lab == i].sum() for i in range(4)], out["runaway"], out["t_max"]


def run(path, n, s0, params=None, procs=4):
    with Pool(procs) as p:
        res = p.map(one, [(path, s, params) for s in range(s0, s0 + n)])
    c = np.array([r[0] for r in res]); parts = np.array([r[1] for r in res]).mean(0)
    return c, parts, sum(r[2] for r in res), max(r[3] for r in res)


if __name__ == "__main__":
    path, n, s0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    params = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    c, parts, nr, tm = run(path, n, s0, params)
    print("%-12s %-40s n=%d mean %.4f med %.4f max %.3f sem %.4f | startup %.4f sp %.4f dist %.4f quiet %.4f | runaway %d tmax %.1f" % (
        path, sys.argv[4] if len(sys.argv) > 4 else "", n, c.mean(), np.median(c), c.max(), c.std() / np.sqrt(n), *parts, nr, tm))
