"""Adversarial safety checks on my model: low setpoint + rich/hot feed (the ignition corner)."""
import numpy as np, simlab
from controller import Controller
N = simlab.N
def run(ctrl, sp_v, caf_f, tf_f, x0, seed=0, noise=1.0):
    rng = np.random.default_rng(seed); x = np.array(x0); c = ctrl; tmax = 0; cost = 0; us = []
    for k in range(N):
        obs = dict(t_min=k*simlab.DT, Ca=x[0]+noise*rng.normal(0, .0018), T=x[1]+noise*rng.normal(0, .19), Ca_sp=sp_v(k))
        u = float(np.clip(c.act(obs), 295, 302)); us.append(u)
        x = simlab.plant_step(x, u, caf_f(k), tf_f(k), simlab.TEXTBOOK); tmax = max(tmax, x[1]); cost += ((x[0]-sp_v(k))/0.01)**2
    return cost/N, tmax, x
class Dumb:
    def __init__(self, guard): self.g = guard
    def act(self, o): return 295.0 if o["T"] > self.g else 302.0
for g in (329, 331, 333, 400):
    print("hold 302 with guard", g, "-> cost %.2f Tmax %.1f final %s" % run(Dumb(g), lambda k: 0.86, lambda k: 1.02, lambda k: 351.5, [0.88, 325.0]))
worst = 0
for seed in range(12):
    r = run(Controller(), lambda k: 0.86, lambda k: 1.02, lambda k: 348.5 if k < 60 else 351.5, [0.90, 322.0], seed)
    r2 = run(Controller(), lambda k: 0.86 if k > 30 else 0.90, lambda k: 0.98 if k < 50 else 1.02, lambda k: 351.5, [0.85, 326.0], seed)
    r3 = run(Controller(dict(t_safe=1e9)), lambda k: 0.86, lambda k: 1.02, lambda k: 348.5 if k < 60 else 351.5, [0.90, 322.0], seed)
    worst = max(worst, r[1], r2[1]); print("seed %d: A cost %.3f Tmax %.1f | B cost %.3f Tmax %.1f | A without guard Tmax %.1f" % (seed, r[0], r[1], r2[0], r2[1], r3[1]))
print("worst Tmax", worst)
