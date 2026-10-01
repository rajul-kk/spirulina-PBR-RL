import sys, json
import numpy as np
import plantsim
path, seed = sys.argv[1], int(sys.argv[2])
params = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
oracle = len(sys.argv) > 4
C = plantsim.load(path)
ctrl = C(params)
c, ra, tmax, rows = plantsim.run(ctrl, seed, oracle=oracle, trace=True)
print("cost", c, "tmax", tmax)
print(" k   yCa     yT     sp      u      Ca      T      Ti     Caf    sq")
for r in rows:
    print("%3d %.4f %7.2f %.4f %7.2f %.4f %7.2f %7.2f %.4f %6.2f" % r)
