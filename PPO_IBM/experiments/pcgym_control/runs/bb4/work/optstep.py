"""Best possible setpoint-step response on the fitted model (offline optimum) vs the MPC's."""
import sys, numpy as np, model
from scipy.optimize import minimize
from controller import Controller
P = model.P_FIT
def traj(u, x0, sp, Caf, Tf, N=40):
    Ca, T = x0; e = []; Ts = []
    for j in range(N):
        e.append((Ca - sp) * 100); Ts.append(T)
        Ca, T = model.step(Ca, T, u[j] if j < len(u) else u[-1], Caf, Tf, P, n=2)
    return np.array(e), np.array(Ts)
def best(x0, sp, Caf, Tf, M=20):
    c = Controller(); uss = np.clip(c._u_ss(sp, Caf, Tf), 295, 302)
    J = lambda u: float(np.sum(traj(np.r_[u, uss], x0, sp, Caf, Tf)[0] ** 2))
    bestr = None
    for init in (np.full(M, uss), np.r_[np.full(4, 295.0), np.full(M - 4, uss)], np.r_[np.full(4, 302.0), np.full(M - 4, uss)],
                 np.r_[np.full(6, 295.0), np.full(3, 302.0), np.full(M - 9, uss)], np.r_[np.full(6, 302.0), np.full(3, 295.0), np.full(M - 9, uss)]):
        r = minimize(J, init, bounds=[(295, 302)] * M, method='L-BFGS-B', options=dict(maxiter=400))
        if bestr is None or r.fun < bestr.fun: bestr = r
    return bestr, uss
def mpc(x0, sp, Caf, Tf, params=None, N=40):
    c = Controller(params); Ca, T = x0; e = []; us = []; Ts = []
    for j in range(N):
        e.append((Ca - sp) * 100); Ts.append(T)
        u = c._mpc([Ca, T, Caf, Tf], sp, c.u_prev if c.u_prev else 298.5); c.u_prev = u; us.append(u)
        Ca, T = model.step(Ca, T, u, Caf, Tf, P, n=2)
    return np.array(e), np.array(us), np.array(Ts)
if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    params = eval(sys.argv[1]) if len(sys.argv) > 1 else None
    for sp0, sp1, Caf, Tf in [(0.87, 0.90, 1.005, 350.8), (0.90, 0.87, 1.005, 350.8), (0.86, 0.90, 1.02, 352.0), (0.90, 0.86, 1.02, 352.0), (0.90, 0.86, 0.99, 349.5), (0.88, 0.89, 1.005, 350.8)]:
        c = Controller(); x0 = model.steady(c._u_ss(sp0, Caf, Tf), Caf, Tf)
        r, uss = best(x0, sp1, Caf, Tf); e, Ts = traj(np.r_[r.x, uss], x0, sp1, Caf, Tf)
        em, um, Tm = mpc(x0, sp1, Caf, Tf, params)
        print('step %.2f->%.2f Caf %.3f Tf %.1f | optimum SSE %.2f (Tmax %.1f) | MPC SSE %.2f (Tmax %.1f) | first sample %.2f' % (sp0, sp1, Caf, Tf, r.fun, Ts.max(), np.sum(em ** 2), Tm.max(), e[0] ** 2))
        print('   u opt', r.x[:14]); print('   u mpc', um[:14]); print('   e opt', e[:14]); print('   e mpc', em[:14])
