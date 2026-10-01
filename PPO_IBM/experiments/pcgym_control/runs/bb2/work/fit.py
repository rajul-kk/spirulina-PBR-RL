"""Fit a first-principles CSTR model to trial logs by EKF innovation likelihood.
Model (time in min):
 dCa/dt = a (Caf - Ca) - k(T) Ca
 dT/dt  = a (Tf - T) + b k(T) Ca + c (Tc - T)
 k(T) = kref * exp(-ER (1/T - 1/Tref)), Tref = 322
Unmeasured disturbances Caf, Tf modelled as random walks.
Reads only runs/bb2/trials/.
"""
import sys, glob, math
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "runs/bb2/work")
from show import load
DT = 26.0 / 120.0
TREF = 322.0
I4 = np.eye(4)


def f(Ca, T, Caf, Tf, u, th):
    a, kref, ER, b, c = th
    k = kref * math.exp(-ER * (1.0 / T - 1.0 / TREF))
    return a * (Caf - Ca) - k * Ca, a * (Tf - T) + b * k * Ca + c * (u - T)


def step(Ca, T, Caf, Tf, u, th):
    h = DT
    k1 = f(Ca, T, Caf, Tf, u, th)
    k2 = f(Ca + 0.5 * h * k1[0], T + 0.5 * h * k1[1], Caf, Tf, u, th)
    k3 = f(Ca + 0.5 * h * k2[0], T + 0.5 * h * k2[1], Caf, Tf, u, th)
    k4 = f(Ca + h * k3[0], T + h * k3[1], Caf, Tf, u, th)
    return (Ca + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            T + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def jac(Ca, T, th):
    a, kref, ER, b, c = th
    k = kref * math.exp(-ER * (1.0 / T - 1.0 / TREF))
    dk = k * ER / (T * T)
    A = np.zeros((4, 4))
    A[0, 0] = -a - k
    A[0, 1] = -dk * Ca
    A[0, 2] = a
    A[1, 0] = b * k
    A[1, 1] = -a + b * dk * Ca - c
    A[1, 3] = a
    Ah = A * DT
    A2 = Ah.dot(Ah)
    return I4 + Ah + A2 / 2.0 + A2.dot(Ah) / 6.0


def ekf(th, d, sCa, sT, qCaf, qTf, x0d, delay=0):
    """Returns (nll, states[n,4], innovations[n-1,2])."""
    R = np.diag([sCa ** 2, sT ** 2])
    Q = np.diag([1e-8, 1e-4, qCaf ** 2, qTf ** 2])
    x = np.array([d["Ca"][0], d["T"][0], x0d[0], x0d[1]])
    P = np.diag([sCa ** 2, sT ** 2, 0.05 ** 2, 5.0 ** 2])
    tot = 0.0
    xs = [x.copy()]
    inn = []
    n = len(d["Ca"])
    for i in range(1, n):
        u = d["Tc"][max(i - 1 - delay, 0)]
        F = jac(x[0], x[1], th)
        Ca, T = step(x[0], x[1], x[2], x[3], u, th)
        x = np.array([Ca, T, x[2], x[3]])
        P = F.dot(P).dot(F.T) + Q
        e = np.array([d["Ca"][i] - Ca, d["T"][i] - T])
        S = P[:2, :2] + R
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        Si = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        if i > 10:
            tot += 0.5 * (e.dot(Si).dot(e) + math.log(det))
        K = P[:, :2].dot(Si)
        x = x + K.dot(e)
        P = P - K.dot(P[:2, :])
        xs.append(x.copy())
        inn.append(e)
    return tot, np.array(xs), np.array(inn)


def unpack(p):
    th = (p[0], math.exp(p[1]), p[2] * 1000, p[3] * 100, p[4])
    return th, math.exp(p[5]), math.exp(p[6]), math.exp(p[7]), math.exp(p[8]), (p[9], p[10] * 100)


P0 = np.array([1.0, math.log(0.11), 8.75, 2.09, 2.09, math.log(0.003), math.log(0.3),
               math.log(0.005), math.log(0.5), 1.0, 3.5])


def total(p, batches, delay=0):
    try:
        th, sCa, sT, qCaf, qTf, x0d = unpack(p)
        v = sum(ekf(th, d, sCa, sT, qCaf, qTf, x0d, delay)[0] for d in batches)
    except Exception:
        return 1e9
    return v if np.isfinite(v) else 1e9


def describe(p):
    th, sCa, sT, qCaf, qTf, x0d = unpack(p)
    return "a=%.4f kref=%.5f ER=%.0f b=%.2f c=%.4f sCa=%.5f sT=%.4f qCaf=%.5f qTf=%.4f Caf0=%.4f Tf0=%.2f" % (
        th + (sCa, sT, qCaf, qTf) + x0d)


if __name__ == "__main__":
    pats = sys.argv[1:] or ["b00[2-7]*"]
    batches = [load(p) for pat in pats for p in sorted(glob.glob("runs/bb2/trials/" + pat))]
    print(len(batches), "batches", flush=True)
    print("nll at textbook-style initial guess: delay0 %.2f delay1 %.2f" % (total(P0, batches, 0), total(P0, batches, 1)), flush=True)
    r = minimize(total, P0, args=(batches,), method="Nelder-Mead",
                 options=dict(maxiter=2500, maxfev=3500, xatol=1e-4, fatol=1e-2, adaptive=True))
    p = r.x
    print("nll", r.fun, "iters", r.nit, flush=True)
    print(describe(p))
    np.save("runs/bb2/work/fit_p.npy", p)
