"""Robustness of a controller file (with its own built-in defaults) in my simulator:
plant = fit2 model, plant = fit3 model, perturbed plants, and a harsh safety scenario."""
import sys
import numpy as np
sys.path.insert(0, "runs/bb2/work")
import sim
Ctrl = sim.load_ctrl(sys.argv[1])
n = int(sys.argv[2])
F2 = (1.0445, 0.11245, 8948.0, 207.54, 2.0758)
F3 = (0.9970, 0.10695, 9012.0, 215.47, 2.1081)
def ev(name, th, **kw):
    r = [sim.run(Ctrl, s, th=th, **kw) for s in range(n)]
    c = np.array([x[0] for x in r])
    print("%-44s mean %.4f median %.4f max %.3f Tmax %.1f" % (name, c.mean(), np.median(c), c.max(), max(x[1] for x in r)), flush=True)
ev("plant=fit2 (Tf 347.5-350.5)", F2)
ev("plant=fit3 (same Tf range: 1.5-2 K low)", F3)
ev("plant=fit2, cooling c +10%", (F2[0], F2[1], F2[2], F2[3], F2[4] * 1.1))
ev("plant=fit2, cooling c -10%", (F2[0], F2[1], F2[2], F2[3], F2[4] * 0.9))
ev("plant=fit2, rate kref +10%", (F2[0], F2[1] * 1.1, F2[2], F2[3], F2[4]))
ev("plant=fit2, heat b +10%", (F2[0], F2[1], F2[2], F2[3] * 1.1, F2[4]))
ev("plant=fit2, noise x2", F2, sCa=0.0044, sT=0.42)
# harsh: hot rich feed, low setpoint, start hot
class Hot:
    pass
x = np.array([0.80, 331.0]); c = Ctrl(); tmax = 0
rng = np.random.RandomState(0)
for i in range(120):
    u = min(max(c.act(dict(t_min=i * sim.DT, Ca=x[0] + 0.0022 * rng.randn(), T=x[1] + 0.21 * rng.randn(), Ca_sp=0.80)), 295.0), 302.0)
    x = sim.plant_step(x, u, (1.06, 356.0), F2); tmax = max(tmax, x[1])
print("harsh scenario (Caf 1.06, Tf 356, sp 0.80, start 331 K): Tmax %.1f final T %.1f Ca %.3f" % (tmax, x[1], x[0]))
