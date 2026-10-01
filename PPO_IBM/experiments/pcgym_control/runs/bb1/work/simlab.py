"""Offline design simulator: my identified model + scenario generator mimicking pilot-batch statistics.
This is NOT the plant - it is my own model, used to tune before spending pilot batches."""
import sys, time, importlib, numpy as np
N = 120; DT = 26.0/120
SIG_CA, SIG_T = 0.0020, 0.20   # plant innovation ML fit on b050-b149 (fit_noise.py)

def rhs(x, tc, caf, tf, m):
    a, k0, ER, beta, alpha = m
    k = k0*np.exp(-ER/x[1])
    return np.array([a*(caf-x[0]) - k*x[0], a*(tf-x[1]) + beta*k*x[0] + alpha*(tc-x[1])])

def plant_step(x, tc, caf, tf, m, nsub=8):
    h = DT/nsub
    for _ in range(nsub):
        k1 = rhs(x, tc, caf, tf, m); k2 = rhs(x+h/2*k1, tc, caf, tf, m)
        k3 = rhs(x+h/2*k2, tc, caf, tf, m); k4 = rhs(x+h*k3, tc, caf, tf, m)
        x = x + h/6*(k1+2*k2+2*k3+k4)
    return x

OPTS = dict(noise=1.0, dist=True, spchange=True, init=True)
def scenario(rng):
    c1 = rng.integers(18, 60); c2 = rng.integers(max(c1+5, 45), 100)
    sp = np.empty(N); v = rng.uniform(0.86, 0.90, 3); sp[:c1] = v[0]; sp[c1:c2] = v[1]; sp[c2:] = v[2]
    # feed disturbances as seen in b020-b049 (EKF replay): levels uniform in a band from the start, 0-2 steps each
    caf = np.full(N, rng.uniform(0.98, 1.02)); tf = np.full(N, rng.uniform(348.5, 351.5))
    for _ in range(rng.choice([0, 1, 1, 2])):
        caf[rng.integers(5, N):] = rng.uniform(0.98, 1.02)
    for _ in range(rng.choice([0, 1, 1, 2])):
        tf[rng.integers(5, N):] = rng.uniform(348.5, 351.5)
    x0 = np.array([rng.uniform(0.848, 0.905), rng.uniform(320.0, 326.0)])  # widened after b010-b019
    if not OPTS["dist"]: caf[:] = caf[0]; tf[:] = tf[0]
    if not OPTS["spchange"]: sp[:] = sp[0]
    if not OPTS["init"]:
        x0[0] = sp[0]
        for _ in range(200): x0 = plant_step(np.array([sp[0], x0[1]]), 0, caf[0], tf[0], (1.0, 7.2e10, 8750.0, 209.2, 0.0))  # T s.t. Ca stays ~sp
    return sp, caf, tf, x0

TEXTBOOK = (1.0, 7.2e10, 8750.0, 209.2, 2.092)
def run_batch(Ctrl, params, seed, m=TEXTBOOK, trace=False):
    rng = np.random.default_rng(seed)
    sp, caf, tf, x = scenario(rng)
    c = Ctrl(params); cost = 0.0; tmax = 0; rows = []
    for k in range(N):
        obs = dict(t_min=k*DT, Ca=x[0] + OPTS['noise']*rng.normal(0, SIG_CA), T=x[1] + OPTS['noise']*rng.normal(0, SIG_T), Ca_sp=sp[k])
        u = float(np.clip(c.act(obs), 295, 302))
        if trace: rows.append((k, x[0], x[1], sp[k], u, caf[k], tf[k], *(c.x if hasattr(c, "x") else ())))
        x = plant_step(x, u, caf[k], tf[k], m)
        # plant scoring convention inferred from b010-b019: state AFTER the move vs the setpoint of that sample
        cost += ((x[0]-sp[k])/0.01)**2; tmax = max(tmax, x[1])
    return (cost/N, tmax, np.array(rows)) if trace else (cost/N, tmax)

def evaluate(modname, params=None, seeds=range(60), m=TEXTBOOK):
    Ctrl = importlib.import_module(modname).Controller
    t0 = time.time(); r = np.array([run_batch(Ctrl, params, s, m) for s in seeds])
    return r[:, 0].mean(), np.median(r[:, 0]), r[:, 0].max(), r[:, 1].max(), (time.time()-t0)/len(r)

if __name__ == "__main__":
    mod = sys.argv[1]; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    print("%s: mean %.3f median %.3f max %.3f Tmax %.1f  (%.2fs/batch)" % ((mod,) + evaluate(mod, None, range(ns))))
