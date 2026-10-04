"""Inspect surrogate scenarios: per-scenario cost and a trace of the worst one."""
import numpy as np, sys
import sim
name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
mod = sim.load_ctrl(name)
rng = np.random.default_rng(0)
scs = [sim.scenario(rng) for _ in range(n)]
res = [sim.run(mod, s, log=True) for s in scs]
c = np.array([r[0] for r in res])
print(np.round(c, 2))
i = int(sys.argv[3]) if len(sys.argv) > 3 else int(np.argmax(c))
print('scenario', i)
for r in res[i][1]:
    print('%3d Ca %.4f T %.2f sp %.4f u %.2f | Caf %.3f Tf %.1f | est %.4f %.2f %.3f %.1f' % tuple(r))
