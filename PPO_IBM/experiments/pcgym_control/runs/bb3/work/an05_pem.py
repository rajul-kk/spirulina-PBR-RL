"""Prediction-error fit of the model parameters through the controller's own EKF (feed shifts are EKF states)."""
import numpy as np, sys
from scipy.optimize import least_squares
from an_common import load
import sim
NAMES = ['a', 'kr', 'ER', 'b', 'c']
def innovations(theta, D, mod, extra):
    P = dict(zip(NAMES, theta)); P.update(extra); out = []
    for d in D:
        c = mod.Controller(P); PP = c.P
        c.x = np.array([d['Ca'][0], d['T'][0], PP['Caf'], PP['Tf']])
        c.Pc = np.diag([PP['r_ca'] ** 2, PP['r_t'] ** 2, 0.05 ** 2, 5.0 ** 2])
        for k in range(1, len(d)):
            c._predict(d['Tc'][k - 1])
            pred = c.x[:2].copy()
            c._update(np.array([d['Ca'][k], d['T'][k]]))
            if k > 3: out.append((np.array([d['Ca'][k], d['T'][k]]) - pred) / np.array([0.002, 0.2]))
    return np.concatenate(out)
if __name__ == '__main__':
    mod = sim.load_ctrl(sys.argv[1]); pats = sys.argv[2].split(',')
    D = [d for p in pats for d in load(p).values()]
    extra = dict(q_caf=0.003, q_tf=0.3, jump_thr=1e9)
    d0 = mod.DEFAULTS; th0 = np.array([d0[n] for n in NAMES])
    r0 = innovations(th0, D, mod, extra); print('batches', len(D), 'start rms', np.sqrt(np.mean(r0 ** 2, )))
    s = least_squares(innovations, th0, args=(D, mod, extra), x_scale=np.abs(th0), loss='soft_l1', diff_step=1e-3)
    r = s.fun; J = s.jac
    cov = np.linalg.inv(J.T @ J) * np.mean(r ** 2)
    print('theta', dict(zip(NAMES, np.round(s.x, 5))))
    print('rel sd', np.round(np.sqrt(np.diag(cov)) / np.abs(s.x), 3), 'rms', np.sqrt(np.mean(r ** 2)))
    rr = r.reshape(-1, 2); print('rms Ca, T innov:', np.sqrt(np.mean(rr ** 2, 0)), 'lag1 autocorr', [np.corrcoef(rr[:-1, i], rr[1:, i])[0, 1] for i in (0, 1)])
