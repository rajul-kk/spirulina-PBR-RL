import sys, json, numpy as np
from multiprocessing import Pool
from simtest import load, run
CF = sys.argv[1]
import os
INOC = [int(x) for x in os.environ.get("INOC","40,120,250,400,1000,3000").split(",")]; MUS = [0.03, 0.04, 0.05]
def job(a):
    cells, mu, params = a
    C = load(CF); h, hs, lost = run(C, cells, mumax=mu, seed=cells, params=params)
    return h
def score(params, pool):
    r = np.array(pool.map(job, [(c, m, params) for c in INOC for m in MUS])).reshape(len(INOC), len(MUS)).mean(1)
    return r
if __name__ == "__main__":
    grid = json.loads(sys.argv[2])
    import itertools
    keys = list(grid)
    with Pool(6) as pool:
        for vals in itertools.product(*[grid[k] for k in keys]):
            p = dict(zip(keys, vals)); r = score(p, pool)
            print(p, round(float(np.log(r).sum()), 3), r.round(0), flush=True)
