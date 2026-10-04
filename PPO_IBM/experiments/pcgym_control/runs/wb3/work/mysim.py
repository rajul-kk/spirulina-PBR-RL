"""My own model of the CSTR (equations read from pcgym/model_classes.py cstr_ode) and of the
scenario generator (distributions read from tasks.py). Does NOT import pcgym or tasks."""
import numpy as np

K0, EA, DH, UAC = 7.2e10, 8750.0, 5e4 / (1000 * 0.239), 5e4 / (1000 * 0.239 * 100)
DT = 26.0 / 120
N = 120


def f(ca, T, Tc, Ti, Caf):
    r = K0 * np.exp(-EA / T) * ca
    return Caf - ca - r, (Ti - T) + DH * r + UAC * (Tc - T)


def step(ca, T, Tc, Ti, Caf, nsub=8):
    h = DT / nsub
    for _ in range(nsub):
        a1, b1 = f(ca, T, Tc, Ti, Caf)
        a2, b2 = f(ca + .5 * h * a1, T + .5 * h * b1, Tc, Ti, Caf)
        a3, b3 = f(ca + .5 * h * a2, T + .5 * h * b2, Tc, Ti, Caf)
        a4, b4 = f(ca + h * a3, T + h * b3, Tc, Ti, Caf)
        ca = ca + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        T = T + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
    return ca, T


def _steps(rng, n, lo, hi, n_changes, min_len=20):
    while True:
        cuts = np.sort(rng.randint(min_len, n - min_len, n_changes))
        if np.all(np.diff(np.concatenate([[0], cuts, [n]])) >= min_len):
            break
    levels = rng.uniform(lo, hi, n_changes + 1)
    return np.repeat(levels, np.diff(np.concatenate([[0], cuts, [n]])))


def scenario(seed):
    rng = np.random.RandomState(seed)
    return {"sp": _steps(rng, N, 0.86, 0.90, 2),
            "Ti": _steps(rng, N, 348.5, 351.5, rng.randint(1, 3)),
            "Caf": _steps(rng, N, 0.98, 1.02, rng.randint(1, 3)),
            "x0": np.array([rng.uniform(0.85, 0.91), rng.uniform(320.0, 326.0)]),
            "noise_seed": int(rng.randint(1 << 30))}


def run(ctrl, seed, perfect=False, trace=False):
    """Simulate one batch. perfect=True gives the controller the true state and disturbances
    (obs gets extra keys) -- used only for lower-bound studies offline."""
    sc = scenario(seed)
    nrng = np.random.RandomState(sc["noise_seed"])
    ca, T = sc["x0"]
    cost, tmax, rows = 0.0, T, []
    for k in range(N):
        obs = {"t_min": k * DT, "Ca": float(ca + nrng.normal(0, 0.002)),
               "T": float(T + nrng.normal(0, 0.2)), "Ca_sp": float(sc["sp"][k])}
        if perfect:
            obs.update(Ca=float(ca), T=float(T), Ti_true=float(sc["Ti"][k]), Caf_true=float(sc["Caf"][k]))
        u = float(np.clip(ctrl.act(obs), 295.0, 302.0))
        ca0, T0 = ca, T
        ca, T = step(ca, T, u, sc["Ti"][k], sc["Caf"][k])
        cost += ((ca - sc["sp"][k]) / 0.01) ** 2
        tmax = max(tmax, T)
        if trace:
            rows.append((k, obs["Ca"], obs["T"], sc["sp"][k], u, ca, T, sc["Ti"][k], sc["Caf"][k], ca0, T0))
    out = {"seed": seed, "cost": cost / N, "runaway": tmax > 335.0, "t_max": tmax}
    return (out, np.array(rows)) if trace else out
