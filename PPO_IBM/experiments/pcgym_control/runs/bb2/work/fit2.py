"""Refit with the initial disturbance guess fixed (Caf0=1, Tf0=350: not identifiable separately
from the random-walk states), then print filtered disturbance trajectories per batch."""
import sys, glob, math
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "runs/bb2/work")
from show import load
import fit

if __name__ == "__main__":
    pats = sys.argv[1:]
    files = [p for pat in pats for p in sorted(glob.glob("runs/bb2/trials/" + pat))]
    batches = [load(p) for p in files]
    print(len(batches), "batches", flush=True)

    def full(q):
        return np.concatenate([q, [1.0, 3.5]])
    q0 = fit.P0[:9]
    prev = np.load("runs/bb2/work/fit_p.npy")
    print("textbook guess nll %.2f ; previous fit with Caf0/Tf0 reset nll %.2f" % (
        fit.total(full(q0), batches), fit.total(full(prev[:9]), batches)), flush=True)
    r = minimize(lambda q: fit.total(full(q), batches), q0, method="Nelder-Mead",
                 options=dict(maxiter=2500, maxfev=3500, xatol=1e-4, fatol=1e-2, adaptive=True))
    p = full(r.x)
    print("nll", r.fun, "iters", r.nit)
    print(fit.describe(p), flush=True)
    np.save("runs/bb2/work/fit2_p.npy", p)
    th, sCa, sT, qCaf, qTf, x0d = fit.unpack(p)
    for name, d in zip(files, batches):
        _, xs, inn = fit.ekf(th, d, sCa, sT, 0.004, 0.4, x0d)
        print(name[-22:], "innov std Ca %.5f T %.3f" % (inn[10:, 0].std(), inn[10:, 1].std()))
        print("  Caf:", " ".join("%.3f" % v for v in xs[5::8, 2]))
        print("  Tf :", " ".join("%.1f" % v for v in xs[5::8, 3]))
