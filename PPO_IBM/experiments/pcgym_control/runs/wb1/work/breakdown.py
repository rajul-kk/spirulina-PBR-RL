"""Where does the cost come from? Attribute each step's squared error to the most recent event."""
import sys, json
from multiprocessing import Pool
import numpy as np
import plantsim

def job(a):
    path, seed, params, oracle = a
    C = plantsim.load(path)
    _, _, _, rows = plantsim.run(C(params), seed, oracle=oracle, trace=True)
    sc = plantsim.scenario(seed)
    sq = np.array([r[-1] for r in rows])
    lab = np.zeros(120, dtype=int)   # 0 start-up, 1 sp, 2 Ti, 3 Caf, 4 quiet(>12 steps after any event)
    last, kind = 0, 0
    for k in range(120):
        if k > 0:
            if sc["sp"][k] != sc["sp"][k-1]: last, kind = k, 1
            if sc["Ti"][k] != sc["Ti"][k-1]: last, kind = k, 2
            if sc["Caf"][k] != sc["Caf"][k-1]: last, kind = k, 3
        lab[k] = kind if k - last <= 12 else 4
    return [sq[lab == i].sum() / 120 for i in range(5)]

if __name__ == "__main__":
    path, n, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    variants = json.loads(sys.argv[4])
    with Pool(5) as pool:
        for name, params, oracle in variants:
            res = np.array(pool.map(job, [(path, s, params, oracle) for s in range(seed0, seed0 + n)]))
            m = res.mean(0)
            print(f"{name:20s} total={m.sum():.4f} startup={m[0]:.4f} sp={m[1]:.4f} Ti={m[2]:.4f} Caf={m[3]:.4f} quiet={m[4]:.4f}", flush=True)
