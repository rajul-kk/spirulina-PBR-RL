import sys, csv, glob
import numpy as np
def load(p):
    with open(p) as f:
        r = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}
if __name__ == "__main__":
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for p in sorted(glob.glob("runs/bb2/trials/" + sys.argv[1])):
        d = load(p)
        print(p, "cost(meas)=%.2f" % np.mean(((d["Ca"]-d["Ca_sp"])/0.01)**2))
        for i in range(0, len(d["step"]), step):
            print("%3d Ca %.4f T %.2f sp %.4f Tc %.2f" % (i, d["Ca"][i], d["T"][i], d["Ca_sp"][i], d["Tc"][i]))
