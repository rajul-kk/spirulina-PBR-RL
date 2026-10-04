"""My own re-implementation of the CSTR plant from the equations in pcgym/model_classes.py and
the scenario description in tasks.py. Used only for offline design; does not import pcgym."""
import numpy as np

N, TSIM = 120, 26.0
DT = TSIM / N
TC_LO, TC_HI = 295.0, 302.0


def rhs(x, Tc, Ti, Caf):
    ca, T = x[..., 0], x[..., 1]
    rA = 7.2e10 * np.exp(-8750.0 / T) * ca
    dca = (Caf - ca) - rA
    dT = (Ti - T) + 5e4 * rA / 239.0 + 5e4 * (Tc - T) / 23900.0
    return np.stack([dca, dT], axis=-1)


def step(x, Tc, Ti, Caf, nsub=8):
    h = DT / nsub
    for _ in range(nsub):
        k1 = rhs(x, Tc, Ti, Caf)
        k2 = rhs(x + 0.5 * h * k1, Tc, Ti, Caf)
        k3 = rhs(x + 0.5 * h * k2, Tc, Ti, Caf)
        k4 = rhs(x + h * k3, Tc, Ti, Caf)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def _steps(rng, n, lo, hi, n_changes, min_len=20):
    while True:
        cuts = np.sort(rng.randint(min_len, n - min_len, n_changes))
        if np.all(np.diff(np.concatenate([[0], cuts, [n]])) >= min_len):
            break
    levels = rng.uniform(lo, hi, n_changes + 1)
    return np.repeat(levels, np.diff(np.concatenate([[0], cuts, [n]])))


def scenario(seed):
    # same distribution as the task (my own seeds; not the plant's scenarios)
    rng = np.random.RandomState(seed + 7_000_000)
    return {"sp": _steps(rng, N, 0.86, 0.90, 2),
            "Ti": _steps(rng, N, 348.5, 351.5, rng.randint(1, 3)),
            "Caf": _steps(rng, N, 0.98, 1.02, rng.randint(1, 3)),
            "x0": np.array([rng.uniform(0.85, 0.91), rng.uniform(320.0, 326.0)]),
            "noise_seed": int(rng.randint(1 << 30))}


def run(ctrl, seed, trace=False):
    sc = scenario(seed)
    x = sc["x0"].copy()
    nr = np.random.RandomState(sc["noise_seed"])
    cost, tmax, rows = 0.0, x[1], []
    for k in range(N):
        obs = {"t_min": k * DT, "Ca": x[0] + nr.normal(0, 0.002), "T": x[1] + nr.normal(0, 0.2),
               "Ca_sp": sc["sp"][k]}
        u = float(np.clip(ctrl.act(dict(obs)), TC_LO, TC_HI))
        x = step(x, u, sc["Ti"][k], sc["Caf"][k])
        e = ((x[0] - sc["sp"][k]) / 0.01) ** 2
        cost += e
        tmax = max(tmax, x[1])
        if trace:
            rows.append((k, obs["Ca"], obs["T"], sc["sp"][k], u, x[0], x[1], sc["Ti"][k], sc["Caf"][k], e))
    return cost / N, tmax > 335, (np.array(rows) if trace else None)


def evaluate(make_ctrl, seeds):
    res = [run(make_ctrl(), s) for s in seeds]
    c = np.array([r[0] for r in res])
    return c, sum(r[1] for r in res)
