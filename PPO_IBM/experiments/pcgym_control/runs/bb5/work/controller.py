"""EKF (Ca, T, feed Ca, feed T) + nonlinear MPC for the jacketed CSTR.

Model identified from pilot batches (first-order exothermic CSTR):
  dCa/dt = q/V (Caf - Ca) - k0 exp(-E/RT) Ca
  dT/dt  = q/V (Tf - T) + J k0 exp(-E/RT) Ca + U (Tc - T)
Feed Ca (Caf) and feed T (Tf) are unmeasured and estimated as random walks.
"""
import math
import numpy as np

QV, K0, ER = 1.0, 7.2e10, 8750.0
J, U = 5e4 / 239.0, 5e4 / 23900.0
DT = 13.0 / 60.0
TC_MIN, TC_MAX = 295.0, 302.0

DEFAULTS = dict(nsub=4, N=15, blocks=(1, 1, 2, 3, 8), lam=0.1, T_lim=331.0, w_T=50.0, T_trip=332.5,
                q=(1e-6, 1e-3, 2e-5, 0.02), r=(0.002 ** 2, 0.2 ** 2), gn_iter=3)


def _deriv(Ca, T, Tc, Caf, Tf):
    k = K0 * math.exp(-ER / min(max(T, 250.0), 420.0))
    return QV * (Caf - Ca) - k * Ca, QV * (Tf - T) + J * k * Ca + U * (Tc - T)


def _step(Ca, T, Tc, Caf, Tf, nsub):
    h = DT / nsub
    for _ in range(nsub):
        a1, b1 = _deriv(Ca, T, Tc, Caf, Tf)
        a2, b2 = _deriv(Ca + 0.5 * h * a1, T + 0.5 * h * b1, Tc, Caf, Tf)
        a3, b3 = _deriv(Ca + 0.5 * h * a2, T + 0.5 * h * b2, Tc, Caf, Tf)
        a4, b4 = _deriv(Ca + h * a3, T + h * b3, Tc, Caf, Tf)
        Ca += h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        T += h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
        Ca = min(max(Ca, 0.0), 2.0)
        T = min(max(T, 250.0), 420.0)
    return Ca, T


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.z = None
        self.P = np.diag([1e-5, 0.05, 4e-4, 1.0])
        self.Q = np.diag(p['q'])
        self.R = np.diag(p['r'])
        self.u_prev = None
        self.plan = None
        bl = list(p['blocks'])
        # map block variables to per-step moves over the horizon
        idx = []
        for b, n in enumerate(bl):
            idx += [b] * n
        idx = (idx + [len(bl) - 1] * p['N'])[:p['N']]
        self.idx = idx
        self.nv = len(bl)

    # ---------------- estimator ----------------
    def _fx(self, z, Tc):
        Ca, T = _step(z[0], z[1], Tc, z[2], z[3], self.p['nsub'])
        return np.array([Ca, T, z[2], z[3]])

    def _ekf(self, y):
        if self.z is None:
            self.z = np.array([y[0], y[1], 1.0, 350.0])
        else:
            z, Tc = self.z, self.u_prev
            z0 = self._fx(z, Tc)
            F = np.zeros((4, 4))
            for j, e in enumerate((1e-5, 1e-3, 1e-5, 1e-3)):
                dz = np.zeros(4)
                dz[j] = e
                F[:, j] = (self._fx(z + dz, Tc) - z0) / e
            self.z = z0
            self.P = F @ self.P @ F.T + self.Q
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.z = self.z + K @ (np.asarray(y) - H @ self.z)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        # keep feed estimates physically sensible
        self.z[2] = min(max(self.z[2], 0.85), 1.15)
        self.z[3] = min(max(self.z[3], 340.0), 360.0)

    # ---------------- MPC ----------------
    def _resid(self, v, sp, u_last):
        p = self.p
        Ca, T, Caf, Tf = self.z
        res = []
        u_old = u_last
        sl = math.sqrt(p['lam'])
        sw = math.sqrt(p['w_T'])
        for k in range(p['N']):
            u = v[self.idx[k]]
            Ca, T = _step(Ca, T, u, Caf, Tf, p['nsub'])
            res.append((Ca - sp) / 0.01)
            res.append(sw * max(0.0, T - p['T_lim']))
            if k == 0 or self.idx[k] != self.idx[k - 1]:
                res.append(sl * (u - u_old))
            u_old = u
        return np.array(res)

    def _solve(self, sp, u_last):
        nv = self.nv
        if self.plan is None:
            v = np.full(nv, u_last)
        else:
            v = np.r_[self.plan[1:], self.plan[-1]][:nv]
        lo, hi = TC_MIN, TC_MAX
        for _ in range(self.p['gn_iter']):
            r0 = self._resid(v, sp, u_last)
            Jm = np.zeros((len(r0), nv))
            for j in range(nv):
                e = 0.01 if v[j] < hi - 0.02 else -0.01
                dv = v.copy()
                dv[j] += e
                Jm[:, j] = (self._resid(dv, sp, u_last) - r0) / e
            # box-constrained Gauss-Newton step via simple active set
            free = np.ones(nv, bool)
            step = np.zeros(nv)
            for _a in range(nv + 1):
                step[:] = 0.0
                fi = np.where(free)[0]
                if len(fi) == 0:
                    break
                A = Jm[:, fi]
                g = A.T @ r0
                Hm = A.T @ A + 1e-6 * np.eye(len(fi))
                step[fi] = -np.linalg.solve(Hm, g)
                new = v + step
                viol = (new < lo - 1e-9) | (new > hi + 1e-9)
                if not viol.any():
                    break
                for j in np.where(viol & free)[0]:
                    free[j] = False
                    step[j] = (lo if new[j] < lo else hi) - v[j]
                # fixed vars move to bound; re-solve with residual updated
                r0 = r0 + Jm[:, ~free] @ step[~free]
                v = v.copy()
                v[~free] = np.clip(v[~free] + step[~free], lo, hi)
                step[~free] = 0.0
            v = np.clip(v + step, lo, hi)
        self.plan = v
        return float(v[0])

    def act(self, obs):
        y = (float(obs['Ca']), float(obs['T']))
        self._ekf(y)
        sp = float(obs['Ca_sp'])
        u_last = self.u_prev if self.u_prev is not None else 299.0
        try:
            u = self._solve(sp, u_last)
        except Exception:
            u = float('nan')
        if not math.isfinite(u):
            # numerical trouble: reset the plan, keep the last move
            self.plan = None
            u = u_last
        # hard safety override on the reactor temperature (runaway limit 335 K)
        T_now = max(y[1], self.z[1])
        if T_now > self.p['T_trip']:
            u = TC_MIN
        elif T_now > self.p['T_lim']:
            u = min(u, u_last - 1.0)
        u = min(max(u, TC_MIN), TC_MAX)
        self.u_prev = u
        return u
