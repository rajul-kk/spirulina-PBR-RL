"""Grey-box CSTR fit by H-step-ahead prediction from measured states.
dCa/dt = a(Caf-Ca) - k Ca ; dT/dt = a(Tf-T) + b k Ca + c(Tc-T); k = kr*exp(-ER(1/T-1/323))"""
import numpy as np, sys
from scipy.optimize import least_squares
from an_common import load
DT = 13.0 / 60.0
def f(x, u, p):
    a, kr, ER, b, c, Caf, Tf = p
    Ca, T = x
    k = kr * np.exp(-ER * (1.0 / T - 1.0 / 323.0))
    return np.array([a * (Caf - Ca) - k * Ca, a * (Tf - T) + b * k * Ca + c * (u - T)])
def step(x, u, p, n=4):
    h = DT / n
    for _ in range(n):
        k1 = f(x, u, p); k2 = f(x + h / 2 * k1, u, p); k3 = f(x + h / 2 * k2, u, p); k4 = f(x + h * k3, u, p)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
def resid(p, D, H):
    r = []
    for d in D:
        Ca, T, u = d['Ca'], d['T'], d['Tc']
        x = np.array([Ca[:-H], T[:-H]])
        for j in range(H):
            x = step(x, u[j:len(u) - H + j], p)
            r.append((x[0] - Ca[j + 1:len(Ca) - H + j + 1]) / 0.002)
            r.append((x[1] - T[j + 1:len(T) - H + j + 1]) / 0.2)
    return np.concatenate(r)
if __name__ == '__main__':
    pat = sys.argv[1] if len(sys.argv) > 1 else 'steps'
    D = list(load(pat).values())
    p0 = np.array([1.0, 0.12, 8000.0, 200.0, 2.0, 1.0, 350.0])
    for H in (1, 3, 6):
        s = least_squares(resid, p0, args=(D, H), x_scale=np.abs(p0), loss='soft_l1')
        r = resid(s.x, D, H)
        print('H', H, 'p', np.array2string(s.x, precision=4), 'rms', np.sqrt(np.mean(r ** 2)))
        J = s.jac; 
        try:
            cov = np.linalg.inv(J.T @ J) * np.mean(s.fun ** 2)
            print('   rel sd', np.array2string(np.sqrt(np.diag(cov)) / np.abs(s.x), precision=3))
        except Exception as e: print(e)
