"""Where does the cost come from? (own-model simulation only)
Runs the controller with (a) normal measurements, (b) noise-free measurements,
(c) oracle state+disturbance knowledge, and splits cost by phase."""
import sys
import numpy as np
sys.path.insert(0, "runs/bb2/work")
import sim
from sweep import FIT

path = sys.argv[1]
n = int(sys.argv[2])
Ctrl = sim.load_ctrl(path)


def run_oracle(seed):
    # replicate sim.run but overwrite the controller's estimate with the truth
    holder = {}

    class Oracle(Ctrl):
        def _estimate(self, y):
            self.x = np.array(holder["x"])
            if self.P is None:
                self.P = np.eye(4)
    rng = np.random.RandomState(seed)
    c = Oracle(dict(FIT))
    x = np.array([rng.uniform(0.855, 0.906), rng.uniform(320, 325.5)])
    t1 = rng.randint(20, 61); t2 = rng.randint(57, 100)
    sps = rng.uniform(0.86, 0.90, 3)
    tC = sorted(rng.randint(3, 118, rng.randint(0, 3)))
    tT = sorted(rng.randint(3, 118, rng.randint(0, 3)))
    vC = rng.uniform(0.975, 1.02, 3); vT = rng.uniform(347.5, 350.5, 3)
    cost = 0.0
    for i in range(120):
        sp = sps[0] if i < t1 else (sps[1] if i < t2 else sps[2])
        d = (vC[sum(1 for t in tC if i >= t)], vT[sum(1 for t in tT if i >= t)])
        cost += ((x[0] - sp) / 0.01) ** 2
        holder["x"] = [x[0], x[1], d[0], d[1]]
        rng.randn(); rng.randn()
        u = min(max(float(c.act(dict(t_min=0, Ca=x[0], T=x[1], Ca_sp=sp))), 295.0), 302.0)
        x = sim.plant_step(x, u, d, sim.TH)
    return cost / 120


a = np.array([sim.run(Ctrl, s, params=dict(FIT))[0] for s in range(n)])
b = np.array([sim.run(Ctrl, s, params=dict(FIT), sCa=1e-9, sT=1e-9)[0] for s in range(n)])
c = np.array([run_oracle(s) for s in range(n)])
print("normal     mean %.4f median %.4f" % (a.mean(), np.median(a)))
print("noise-free mean %.4f median %.4f" % (b.mean(), np.median(b)))
print("oracle     mean %.4f median %.4f" % (c.mean(), np.median(c)))
