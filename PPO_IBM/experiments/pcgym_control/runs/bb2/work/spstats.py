import sys, glob
import numpy as np
sys.path.insert(0, "runs/bb2/work")
from show import load
for p in sorted(glob.glob("runs/bb2/trials/" + (sys.argv[1] if len(sys.argv) > 1 else "b*.csv"))):
    d = load(p)
    sp = d["Ca_sp"]; ch = [0] + [i for i in range(1, len(sp)) if sp[i] != sp[i-1]]
    print(p[-22:], "Ca0 %.4f T0 %.2f" % (d["Ca"][0], d["T"][0]), " ".join("%d:%.4f" % (i, sp[i]) for i in ch))
