"""Is the MPC close to the best achievable setpoint-step response? (own-model simulation only)
Noise-free, known disturbance. Compare the MPC with a brute-force bang-bang-then-hold family:
n1 samples at one limit, one intermediate sample, n2 at the other limit, one intermediate, then
the steady-state input."""
import sys, itertools
import numpy as np
sys.path.insert(0, "runs/bb2/work")
import sim
from sweep import FIT

Ctrl = sim.load_ctrl(sys.argv[1])
th = sim.TH
d = (1.0, 349.0)


def steady(sp):
    a, kref, ER, b, c = th
    ks = a * (d[0] - sp) / sp
    Ts = 1.0 / (1.0 / 322.0 - np.log(ks / kref) / ER)
    return Ts, Ts - (a * (d[1] - Ts) + b * ks * sp) / c


def cost_seq(x, useq, uss, sp, n=40):
    J = 0.0
    for j in range(n):
        u = useq[j] if j < len(useq) else uss
        x = sim.plant_step(x, u, d, th)
        J += ((x[0] - sp) / 0.01) ** 2
    return J


for sp0, sp1 in [(0.87, 0.90), (0.90, 0.87), (0.875, 0.885), (0.865, 0.90), (0.90, 0.862)]:
    T0, u0 = steady(sp0)
    T1, u1 = steady(sp1)
    x0 = np.array([sp0, T0])
    # MPC with oracle knowledge
    for kw in ({}, {"N": 20, "M": 8, "iters": 10}):
        prm = dict(FIT); prm.update(kw)
        c = Ctrl(prm)
        c.x = np.array([x0[0], x0[1], d[0], d[1]]); c.P = np.eye(4) * 1e-12
        c.Q = c.Q * 0; c.u_prev = u0
        x = x0.copy(); J = 0.0; us = []
        for j in range(40):
            c.x = np.array([x[0], x[1], d[0], d[1]])
            u = min(max(c._mpc(sp1), 295.0), 302.0)
            us.append(u)
            x = sim.plant_step(x, u, d, th)
            J += ((x[0] - sp1) / 0.01) ** 2
        print("sp %.3f->%.3f (uss %.2f->%.2f) MPC %s: J=%.3f  u: %s" % (sp0, sp1, u0, u1, kw, J, " ".join("%.1f" % v for v in us[:10])))
    lo, hi = (295.0, 302.0) if sp1 > sp0 else (302.0, 295.0)
    best = (1e9, None)
    grid = np.linspace(295, 302, 8)
    for n1 in range(0, 9):
        for n2 in range(0, 7):
            for m1 in grid:
                for m2 in grid:
                    useq = [lo] * n1 + [m1] + [hi] * n2 + [m2]
                    J = cost_seq(x0, useq, u1, sp1)
                    if J < best[0]:
                        best = (J, (n1, m1, n2, m2))
    print("   brute-force bang-bang family: J=%.3f  %s   (cost units per batch: /120 = %.4f)" % (best[0], best[1], best[0] / 120))
