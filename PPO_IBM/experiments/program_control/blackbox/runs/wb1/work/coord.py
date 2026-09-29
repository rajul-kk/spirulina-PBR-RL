import sys, json, os, numpy as np
from multiprocessing import Pool
import tune
if __name__ == "__main__":
    grid = json.loads(sys.argv[2]); th = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    with Pool(6) as pool:
        base = tune.score(th, pool); print("base", round(float(np.log(base).sum()),4), base.round(0), flush=True)
        for key, vals in grid.items():
            res = []
            for v in vals:
                t2 = dict(th); t2[key] = v; s = tune.score(t2, pool); res.append((float(np.log(s).sum()), v, s))
            b = max(res, key=lambda x: x[0]); th[key] = b[1]
            print(key, [(v, round(x, 4)) for x, v, _ in res], "->", b[1], b[2].round(0), flush=True)
    print(json.dumps(th))
