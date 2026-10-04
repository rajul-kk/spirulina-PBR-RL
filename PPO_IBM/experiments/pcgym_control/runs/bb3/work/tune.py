"""Compare parameter variants on the surrogate. usage: tune.py ctrl n '[{...},{...}]' [seed]"""
import sys, json, time, numpy as np, sim
name, n = sys.argv[1], int(sys.argv[2])
variants = json.loads(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
mod = sim.load_ctrl(name)
for v in variants:
    t = time.time(); rng = np.random.default_rng(seed); c = []; du = []
    for _ in range(n):
        j, rows = sim.run(mod, sim.scenario(rng), v, log=True); c.append(j); du.append(np.std(np.diff(rows[:, 4])))
    c = np.array(c)
    print('%-75s mean %.4f med %.3f max %.2f sd(dTc) %.2f (%.0fs)' % (json.dumps(v), c.mean(), np.median(c), c.max(), np.mean(du), time.time() - t), flush=True)
