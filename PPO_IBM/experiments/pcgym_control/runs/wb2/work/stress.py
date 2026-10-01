"""Own-model stress tests of a controller: corner scenarios (hot/rich feed, extreme start states,
largest setpoint and feed steps at the closest allowed spacing), act() timing and behaviour on a
bad measurement. No pilot batches used.

usage: python stress.py <controller.py>
"""
import itertools
import sys
import time

import numpy as np

import sim

C = sim.load(sys.argv[1]).Controller


def run(sp, Ti, Caf, x0, seed=0, bad=None):
    rng = np.random.RandomState(seed)
    ctrl = C()
    ca, T = x0
    sq = np.zeros(sim.N); tmax = T; tact = 0.0
    for k in range(sim.N):
        obs = {"t_min": k * sim.DT, "Ca": ca + rng.normal(0, 0.002), "T": T + rng.normal(0, 0.2),
               "Ca_sp": float(sp[k])}
        if bad is not None and k == bad:
            obs["Ca"] = float("nan")
        t0 = time.perf_counter()
        u = float(ctrl.act(obs))
        tact = max(tact, time.perf_counter() - t0)
        assert np.isfinite(u) and 295.0 <= u <= 302.0, u
        ca, T = sim.plant_step(ca, T, u, float(Ti[k]), float(Caf[k]))
        sq[k] = ((ca - sp[k]) / 0.01) ** 2
        tmax = max(tmax, T)
    return sq.mean(), tmax, tact


def pw(levels, cuts=(40, 80)):
    out = np.empty(sim.N)
    edges = [0, *cuts, sim.N]
    for i, lv in enumerate(levels):
        out[edges[i]:edges[i + 1]] = lv
    return out


worst = (0.0, 0.0, None)
tact = 0.0
n = 0
for sps, tis, cafs, ca0, T0 in itertools.product(
        [(0.86, 0.90, 0.86), (0.90, 0.86, 0.90), (0.86, 0.86, 0.86)],
        [(348.5, 351.5, 348.5), (351.5, 348.5, 351.5), (351.5, 351.5, 351.5)],
        [(0.98, 1.02, 0.98), (1.02, 0.98, 1.02), (1.02, 1.02, 1.02)],
        [0.85, 0.91], [320.0, 326.0]):
    c, tm, ta = run(pw(sps), pw(tis, (30, 70)), pw(cafs, (50, 90)), (ca0, T0), seed=n)
    n += 1
    tact = max(tact, ta)
    if tm > worst[1]:
        worst = (c, tm, (sps, tis, cafs, ca0, T0))
print("corner scenarios: %d, highest T %.2f K (cost %.3f) in %s" % (n, worst[1], worst[0], worst[2]))
print("slowest act() call: %.1f ms" % (1e3 * tact))
c, tm, _ = run(pw((0.88, 0.87, 0.89)), pw((350, 351, 349)), pw((1.0, 1.01, 0.99)), (0.88, 323.0), bad=30)
print("NaN measurement at sample 30: cost %.3f, max T %.2f" % (c, tm))
