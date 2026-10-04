"""Grey-box fit of a 2-state CSTR by one-step-ahead prediction from measured states."""
import sys, glob, numpy as np
from scipy.optimize import least_squares
DT = 26.0 / 120
TREF = 322.0
def f(Ca, T, Tc, p, Caf, Tf):
    a, lnk, E, B, c = p
    k = np.exp(lnk - E * (1.0 / T - 1.0 / TREF))
    return a * (Caf - Ca) - k * Ca, a * (Tf - T) + B * k * Ca + c * (Tc - T)
def step(Ca, T, Tc, p, Caf, Tf, n=4):
    h = DT / n
    for _ in range(n):
        k1 = f(Ca, T, Tc, p, Caf, Tf)
        k2 = f(Ca + h/2*k1[0], T + h/2*k1[1], Tc, p, Caf, Tf)
        k3 = f(Ca + h/2*k2[0], T + h/2*k2[1], Tc, p, Caf, Tf)
        k4 = f(Ca + h*k3[0], T + h*k3[1], Tc, p, Caf, Tf)
        Ca = Ca + h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        T = T + h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    return Ca, T
def load(pats):
    D = []
    for pat in pats:
        for fn in sorted(glob.glob('runs/bb4/trials/' + pat)):
            d = np.genfromtxt(fn, delimiter=',', names=True)
            D.append((fn, d))
    return D
def resid(theta, D, sep=False):
    p = theta[:5]; Caf, Tf = theta[5], theta[6]
    out = []
    for fn, d in D:
        Ca1, T1 = step(d['Ca'][:-1], d['T'][:-1], d['Tc'][:-1], p, Caf, Tf)
        eC = (d['Ca'][1:] - Ca1) / 0.004; eT = (d['T'][1:] - T1) / 0.3
        out.append((eC, eT) if sep else np.concatenate([eC, eT]))
    return out if sep else np.concatenate(out)
if __name__ == '__main__':
    D = load(sys.argv[1:])
    th0 = np.array([1.0, np.log(0.1), 8000.0, 200.0, 2.0, 1.0, 350.0])
    r = least_squares(resid, th0, args=(D,), x_scale=[1, 1, 1000, 100, 1, 1, 100], loss='soft_l1')
    np.set_printoptions(precision=4, suppress=True)
    print('a lnk E B c Caf Tf =', r.x, ' kref=', np.exp(r.x[1]))
    J = r.jac; cov = np.linalg.inv(J.T @ J); print('sd', np.sqrt(np.diag(cov)))
    for (fn, d), (eC, eT) in zip(D, resid(r.x, D, sep=True)):
        print(fn, 'rms eC %.4f eT %.3f' % (np.std(eC) * 0.004, np.std(eT) * 0.3))
        print('  eC blockmeans(x1e3)', np.round([eC[i:i+10].mean() * 4 for i in range(0, 119, 10)], 1))
        print('  eT blockmeans     ', np.round([eT[i:i+10].mean() * 0.3 for i in range(0, 119, 10)], 2))
