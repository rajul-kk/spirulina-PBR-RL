"""Rebuild the scenario of each logged pilot batch (setpoints, feed, start state, noise sequence)
from the privileged CSV and replay it on my own model with any controller. Also prints the phase
split of the logged cost and scenario statistics.

usage: python replay.py [controller.py] [--glob '../trials/b*.csv'] [--params '{...}']
"""
import argparse
import csv
import glob
import json

import numpy as np

import sim


def read(fn):
    rows = list(csv.DictReader(open(fn)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def back_x0(d):
    """Start state such that one plant step with Tc[0] gives the logged true state."""
    x = np.array([d["Ca"][0], d["T"][0]])
    tgt = np.array([d["Ca_true"][0], d["T_true"][0]])
    for _ in range(20):
        f0 = np.array(sim.plant_step(x[0], x[1], d["Tc"][0], d["Ti"][0], d["Caf"][0]))
        J = np.zeros((2, 2))
        for i, h in enumerate((1e-6, 1e-4)):
            xp = x.copy(); xp[i] += h
            J[:, i] = (np.array(sim.plant_step(xp[0], xp[1], d["Tc"][0], d["Ti"][0], d["Caf"][0])) - f0) / h
        x = x - np.linalg.solve(J, f0 - tgt)
    return x


def scenario_from_log(d):
    x0 = back_x0(d)
    prev_ca = np.concatenate([[x0[0]], d["Ca_true"][:-1]])
    prev_T = np.concatenate([[x0[1]], d["T_true"][:-1]])
    return {"sp": d["Ca_sp"], "Ti": d["Ti"], "Caf": d["Caf"], "x0": x0,
            "nca": d["Ca"] - prev_ca, "nT": d["T"] - prev_T}


def run(Controller, sc, params=None):
    ctrl = Controller(params)
    ca, T = float(sc["x0"][0]), float(sc["x0"][1])
    n = len(sc["sp"])
    sq = np.zeros(n); tmax = T
    for k in range(n):
        obs = {"t_min": k * sim.DT, "Ca": ca + sc["nca"][k], "T": T + sc["nT"][k], "Ca_sp": float(sc["sp"][k])}
        if hasattr(ctrl, "truth"):
            ctrl.truth = (ca, T, float(sc["Ti"][k]), float(sc["Caf"][k]))
        u = min(max(float(ctrl.act(obs)), 295.0), 302.0)
        ca, T = sim.plant_step(ca, T, u, float(sc["Ti"][k]), float(sc["Caf"][k]))
        sq[k] = ((ca - sc["sp"][k]) / 0.01) ** 2
        tmax = max(tmax, T)
    return sq, tmax


def phases(sc, sq, w=10):
    n = len(sq)
    lab = np.zeros(n, dtype=int)
    for name, code in (("Ti", 3), ("Caf", 3), ("sp", 2)):
        for c in np.nonzero(np.diff(sc[name]))[0] + 1:
            lab[c:c + w] = code
    lab[:w] = 1
    return [float(sq[lab == c].sum() / n) for c in (1, 2, 3, 0)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ctrl", nargs="?")
    ap.add_argument("--glob", default="../trials/b*.csv")
    ap.add_argument("--params", default="{}")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    C = sim.load(a.ctrl).Controller if a.ctrl else None
    logged, replayed, ph, stats = [], [], [], []
    for fn in sorted(glob.glob(a.glob)):
        d = read(fn)
        sc = scenario_from_log(d)
        sq_log = ((d["Ca_true"] - d["Ca_sp"]) / 0.01) ** 2
        logged.append(sq_log.mean()); ph.append(phases(sc, sq_log))
        stats.append([sc["x0"][0], sc["x0"][1], sc["sp"][0], sc["x0"][0] - sc["sp"][0],
                      len(np.nonzero(np.diff(sc["sp"]))[0]), len(np.nonzero(np.diff(sc["Ti"]))[0]),
                      len(np.nonzero(np.diff(sc["Caf"]))[0])])
        line = "%s logged %.4f" % (fn[-13:-4], logged[-1])
        if C:
            sq, tmax = run(C, sc, json.loads(a.params))
            replayed.append(sq.mean())
            line += "  replay %.4f  Tmax %.1f" % (replayed[-1], tmax)
        if a.v:
            print(line, " x0 %.4f %.2f sp0 %.4f" % (sc["x0"][0], sc["x0"][1], sc["sp"][0]))
    logged = np.array(logged); ph = np.array(ph); stats = np.array(stats)
    print("n %d logged mean %.4f median %.4f" % (len(logged), logged.mean(), np.median(logged)))
    print("logged phases: start %.4f sp %.4f feed %.4f quiet %.4f" % tuple(ph.mean(axis=0)))
    if C:
        r = np.array(replayed)
        print("replay mean %.4f median %.4f  (paired diff %.4f +/- %.4f)" % (
            r.mean(), np.median(r), (r - logged).mean(), (r - logged).std() / np.sqrt(len(r))))
    print("x0 Ca: mean %.4f min %.4f max %.4f | T0 mean %.2f min %.2f max %.2f" % (
        stats[:, 0].mean(), stats[:, 0].min(), stats[:, 0].max(), stats[:, 1].mean(), stats[:, 1].min(), stats[:, 1].max()))
    print("sp0 mean %.4f | mean sq of start error (units) %.3f | changes sp %.2f Ti %.2f Caf %.2f" % (
        stats[:, 2].mean(), np.mean((stats[:, 3] / 0.01) ** 2), stats[:, 4].mean(), stats[:, 5].mean(), stats[:, 6].mean()))
