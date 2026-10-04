"""My own surrogate of the pilot plant, built from the identified model, for tuning before spending real batches.
Reads nothing outside runs/bb3/work."""
import numpy as np, importlib.util, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))


def load_ctrl(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TRUE = dict(a=0.834, kr=0.1048, ER=9183.0, b=216.3, c=2.033, Caf=0.9948, Tf=357.6, Tref=323.0)
_step = load_ctrl('c03_mpc').step


def scenario(rng, true=TRUE):
    sp = rng.uniform(0.86, 0.90, 3)
    t1 = rng.integers(24, 60)
    t2 = rng.integers(t1 + 15, 100)
    spv = np.r_[np.full(t1, sp[0]), np.full(t2 - t1, sp[1]), np.full(120 - t2, sp[2])]
    Caf = np.full(120, true['Caf'] + rng.uniform(-0.015, 0.02))
    Tf = np.full(120, true['Tf'] + rng.uniform(-1, 1))
    for _ in range(rng.integers(0, 3)):
        k = rng.integers(5, 118)
        Caf[k:] = true['Caf'] + rng.uniform(-0.03, 0.03)
    for _ in range(rng.integers(0, 3)):
        k = rng.integers(5, 118)
        Tf[k:] = true['Tf'] + rng.uniform(-2.5, 2.5)
    return dict(sp=spv, Caf=Caf, Tf=Tf, Ca0=rng.uniform(0.85, 0.90), T0=rng.uniform(320, 326),
                seed=int(rng.integers(1 << 30)))


def run(mod, sc, params=None, true=TRUE, log=False):
    rng = np.random.default_rng(sc['seed'])
    c = mod.Controller(params)
    Ca, T = sc['Ca0'], sc['T0']
    J = 0.0
    rows = []
    for k in range(120):
        obs = dict(t_min=k * 13 / 60., Ca=Ca + 0.002 * rng.standard_normal(), T=T + 0.2 * rng.standard_normal(),
                   Ca_sp=sc['sp'][k])
        u = float(np.clip(c.act(obs), 295, 302))
        J += ((Ca - sc['sp'][k]) / 0.01) ** 2
        if log:
            rows.append((k, Ca, T, sc['sp'][k], u, sc['Caf'][k], sc['Tf'][k]) + tuple(getattr(c, 'x', [0, 0, 0, 0])))
        Ca, T = _step(Ca, T, u, sc['Caf'][k], sc['Tf'][k], true, n=4)
    return (J / 120, np.array(rows)) if log else J / 120


def evaluate(name, params=None, n=40, seed=0, true=TRUE):
    mod = load_ctrl(name)
    rng = np.random.default_rng(seed)
    return np.array([run(mod, scenario(rng, true), params, true) for _ in range(n)])


if __name__ == '__main__':
    t = time.time()
    c = evaluate(sys.argv[1], n=int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    print('mean %.3f median %.3f max %.3f  (%.1fs)' % (c.mean(), np.median(c), c.max(), time.time() - t))
