"""Closed-loop tests of a controller on the fitted model (tuning aid only; the plant is the judge)."""
import sys, time, importlib, numpy as np
import model


def _levels(rng, lo, hi):
    n = rng.choice([0, 1, 2], p=[0.45, 0.45, 0.10]); v = np.full(120, rng.uniform(lo, hi))
    for t in sorted(rng.integers(15, 105, n)): v[t:] = rng.uniform(lo, hi)
    return v


def scenario(rng):
    """Scenario statistics taken from pilot batches 0-11 (see dist.py output in the notebook)."""
    t1 = rng.integers(20, 72); t2 = int(np.clip(t1 + rng.integers(27, 70), 69, 100))
    sps = rng.uniform(0.86, 0.90, 3)
    sp = np.r_[np.full(t1, sps[0]), np.full(t2 - t1, sps[1]), np.full(120 - t2, sps[2])]
    return dict(sp=sp, Caf=_levels(rng, 0.985, 1.027), Tf=_levels(rng, 349.2, 352.5),
                x0=(rng.uniform(0.85, 0.907), rng.uniform(320, 325.5)))


def run(Ctrl, params, sc, rng, pp=model.P_FIT, log=False):
    c = Ctrl(params); Ca, T = sc['x0']; cost = 0.0; tmax = T; L = []
    for k in range(120):
        obs = dict(t_min=k * model.DT, Ca=Ca + rng.normal(0, 0.002), T=T + rng.normal(0, 0.2), Ca_sp=sc['sp'][k])
        u = min(max(c.act(obs), 295.0), 302.0)
        cost += ((Ca - sc['sp'][k]) / 0.01) ** 2
        if log: L.append((k, Ca, T, sc['sp'][k], u, c.x[2], sc['Caf'][k], c.x[3], sc['Tf'][k]))
        Ca, T = model.step(Ca, T, u, sc['Caf'][k], sc['Tf'][k], pp); tmax = max(tmax, T)
    return (cost / 120, tmax, L) if log else (cost / 120, tmax)


def evaluate(modname, params=None, n=20, seed=0, pp=model.P_FIT):
    Ctrl = importlib.import_module(modname).Controller
    costs = []; tm = 0
    for i in range(n):
        rng = np.random.default_rng(seed * 1000 + i); sc = scenario(rng)
        c, t = run(Ctrl, params, sc, rng, pp); costs.append(c); tm = max(tm, t)
    return np.mean(costs), np.median(costs), np.max(costs), tm


if __name__ == '__main__':
    t0 = time.time()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    params = eval(sys.argv[3]) if len(sys.argv) > 3 else None
    print('mean %.3f median %.3f max %.3f Tmax %.1f' % evaluate(sys.argv[1], params, n), 'time %.1fs' % (time.time() - t0))
