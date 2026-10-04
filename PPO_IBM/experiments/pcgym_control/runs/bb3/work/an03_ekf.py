"""Replay an augmented-state EKF [Ca,T,Caf,Tf] over logs to see the unmeasured feed shifts."""
import numpy as np, sys
from an_common import load
from an02_fit import step
P0 = np.array([0.834, 0.1048, 9183., 216.3, 2.033, 0.9948, 357.6])
def fx(x, u, p):
    pp = p.copy(); pp[5] = x[2]; pp[6] = x[3]
    y = step(x[:2], u, pp, n=2)
    return np.array([y[0], y[1], x[2], x[3]])
def ekf(d, p, qd=(0.004, 0.4)):
    x = np.array([d['Ca'][0], d['T'][0], p[5], p[6]])
    P = np.diag([0.002 ** 2, 0.2 ** 2, 0.03 ** 2, 3.0 ** 2])
    Q = np.diag([1e-8, 1e-4, qd[0] ** 2, qd[1] ** 2]); R = np.diag([0.002 ** 2, 0.2 ** 2])
    H = np.array([[1., 0, 0, 0], [0, 1., 0, 0]]); out = []; innov = []
    eps = np.array([1e-5, 1e-3, 1e-5, 1e-3])
    for k in range(len(d)):
        if k > 0:
            F = np.zeros((4, 4)); u = d['Tc'][k - 1]
            for i in range(4):
                e = np.zeros(4); e[i] = eps[i]
                F[:, i] = (fx(x + e, u, p) - fx(x - e, u, p)) / (2 * eps[i])
            x = fx(x, u, p); P = F @ P @ F.T + Q
            y = np.array([d['Ca'][k], d['T'][k]]) - x[:2]
            S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
            innov.append(y / np.sqrt(np.diag(S)))
            x = x + K @ y; P = (np.eye(4) - K @ H) @ P
        out.append(x.copy())
    return np.array(out), np.array(innov)
if __name__ == '__main__':
    D = load(sys.argv[1] if len(sys.argv) > 1 else '*')
    for k, d in D.items():
        X, I = ekf(d, P0)
        print(k, 'innov rms', np.round(np.sqrt(np.mean(I ** 2, 0)), 2))
        print('  Caf', ' '.join('%.3f' % v for v in X[5::6, 2]))
        print('  Tf ', ' '.join('%.1f' % v for v in X[5::6, 3]))
