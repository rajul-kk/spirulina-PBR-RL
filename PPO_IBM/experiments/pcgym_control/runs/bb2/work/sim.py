"""Offline simulator built ONLY from my own fitted model, for tuning before spending plant
batches. The scenario generator mimics what was seen in the pilot logs."""
import sys, importlib.util, time
import numpy as np
DT = 26.0 / 120.0
TH = (1.0445, 0.11245, 8948.0, 207.54, 2.0758)  # fit2 result


def load_ctrl(path):
    spec = importlib.util.spec_from_file_location("ctrl_" + str(abs(hash(path))), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Controller


def plant_f(x, u, d, th):
    a, kref, ER, b, c = th
    k = kref * np.exp(-ER * (1 / x[1] - 1 / 322.0))
    return np.array([a * (d[0] - x[0]) - k * x[0], a * (d[1] - x[1]) + b * k * x[0] + c * (u - x[1])])


def plant_step(x, u, d, th, n=4):
    h = DT / n
    for _ in range(n):
        k1 = plant_f(x, u, d, th)
        k2 = plant_f(x + h / 2 * k1, u, d, th)
        k3 = plant_f(x + h / 2 * k2, u, d, th)
        k4 = plant_f(x + h * k3, u, d, th)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def run(Ctrl, seed, th=TH, params=None, sCa=0.0022, sT=0.21, trace=False):
    """Scenario statistics copied from what the pilot logs showed (b000-b012):
    start-up Ca 0.855-0.906, T 320-325.5; setpoints 0.86-0.90 with two changes;
    feed Caf 0.975-1.02 and Tf 347.5-350.5 (model units) with 0-2 step shifts each."""
    rng = np.random.RandomState(seed)
    c = Ctrl(params)
    x = np.array([rng.uniform(0.855, 0.906), rng.uniform(320, 325.5)])
    t1 = rng.randint(20, 61)
    t2 = rng.randint(57, 100)
    sps = rng.uniform(0.86, 0.90, 3)
    tC = sorted(rng.randint(3, 118, rng.randint(0, 3)))
    tT = sorted(rng.randint(3, 118, rng.randint(0, 3)))
    vC = rng.uniform(0.975, 1.02, 3)
    vT = rng.uniform(347.5, 350.5, 3)
    cost = 0.0
    tmax = 0.0
    tr = []
    for i in range(120):
        sp = sps[0] if i < t1 else (sps[1] if i < t2 else sps[2])
        d = (vC[sum(1 for t in tC if i >= t)], vT[sum(1 for t in tT if i >= t)])
        cost += ((x[0] - sp) / 0.01) ** 2
        obs = dict(t_min=i * DT, Ca=x[0] + sCa * rng.randn(), T=x[1] + sT * rng.randn(), Ca_sp=sp)
        u = min(max(float(c.act(obs)), 295.0), 302.0)
        if trace:
            tr.append((i, x[0], x[1], sp, u, d[0], d[1]))
        x = plant_step(x, u, d, th)
        tmax = max(tmax, x[1])
    if trace:
        return cost / 120, tmax, tr
    return cost / 120, tmax


def evaluate(Ctrl, n=30, params=None, **kw):
    res = [run(Ctrl, s, params=params, **kw) for s in range(n)]
    cs = np.array([r[0] for r in res])
    return cs.mean(), np.median(cs), cs.max(), max(r[1] for r in res)


if __name__ == "__main__":
    Ctrl = load_ctrl(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    t0 = time.time()
    m, md, mx, tm = evaluate(Ctrl, n)
    print("mean %.3f median %.3f max %.3f  Tmax %.1f  (%.1fs/batch)" % (m, md, mx, tm, (time.time() - t0) / n))
