import json, sys, time
from multiprocessing import Pool
from opt import evaluate, summary
import opt
if __name__ == "__main__":
    base = json.loads(sys.argv[1]); grid = json.loads(sys.argv[2])
    if len(sys.argv) > 3: opt.CTRL = sys.argv[3]
    with Pool(4) as pool:
        print("base", "wmean %.0f p25 %.0f lost %d | %s" % summary(evaluate(base, ctrl=opt.CTRL, pool=pool)), flush=True)
        for k, vals in grid.items():
            for v in vals:
                p = dict(base); p[k] = v
                print(k, v, "wmean %.0f p25 %.0f lost %d | %s" % summary(evaluate(p, ctrl=opt.CTRL, pool=pool)), flush=True)
