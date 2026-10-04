"""Jacketed CSTR concentration controller (run bb4).

Extended Kalman filter on a fitted 2-state reactor model augmented with the two unmeasured feed
disturbances (feed concentration, feed temperature), followed by a short-horizon nonlinear
model-predictive controller with actuator limits and a reactor-temperature ceiling.
All model numbers were fitted from pilot-plant batches (see LAB_NOTEBOOK.md).
"""
import math
import numpy as np

DT = 26.0 / 120.0      # sample time, min
TREF = 322.0           # reference temperature for the rate constant, K
U_MIN, U_MAX = 295.0, 302.0

DEFAULTS = dict(
    # plant model: dCa/dt = a (Caf - Ca) - k Ca ; dT/dt = a (Tf - T) + B k Ca + c (Tc - T)
    a=0.9697, lnk=-2.1531, E=8604.2, B=203.33, c=2.0808,
    Caf0=1.005, Tf0=350.7,
    # EKF
    r_ca=0.002, r_t=0.2,                 # measurement noise sd
    q_ca=0.0004, q_t=0.04,               # process noise sd per sample (states)
    q_caf=0.0020, q_tf=0.20,             # random-walk sd per sample (disturbances)
    p0_caf=0.012, p0_tf=1.5,             # initial disturbance uncertainty
    # MPC
    H=8, P=14, lam=0.002, t_max=331.0, w_t=4.0, iters=3,
)


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.a, self.lnk, self.E, self.B, self.c = p['a'], p['lnk'], p['E'], p['B'], p['c']
        self.x = None
        self.Pm = None
        self.u_prev = None
        self.H = int(p['H'])
        self.P = int(p['P'])
        self.useq = None
        self.Q = np.diag([p['q_ca'] ** 2, p['q_t'] ** 2, p['q_caf'] ** 2, p['q_tf'] ** 2])
        self.R = np.diag([p['r_ca'] ** 2, p['r_t'] ** 2])

    # ---------------- model ----------------
    def _f(self, Ca, T, Tc, Caf, Tf):
        k = math.exp(self.lnk - self.E * (1.0 / T - 1.0 / TREF))
        return self.a * (Caf - Ca) - k * Ca, self.a * (Tf - T) + self.B * k * Ca + self.c * (Tc - T)

    def _step(self, Ca, T, Tc, Caf, Tf, n=1):
        h = DT / n
        f = self._f
        for _ in range(n):
            a1, b1 = f(Ca, T, Tc, Caf, Tf)
            a2, b2 = f(Ca + 0.5 * h * a1, T + 0.5 * h * b1, Tc, Caf, Tf)
            a3, b3 = f(Ca + 0.5 * h * a2, T + 0.5 * h * b2, Tc, Caf, Tf)
            a4, b4 = f(Ca + h * a3, T + h * b3, Tc, Caf, Tf)
            Ca += h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
            T += h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
            if T > 420.0:
                T = 420.0
            if T < 250.0:
                T = 250.0
            if Ca < 1e-4:
                Ca = 1e-4
        return Ca, T

    def _u_ss(self, sp, Caf, Tf):
        """Jacket temperature that holds Ca = sp at steady state for the estimated feed."""
        sp = min(max(sp, 0.05), Caf - 1e-3)
        k = self.a * (Caf - sp) / sp
        T = 1.0 / (1.0 / TREF - (math.log(k) - self.lnk) / self.E)
        return T - (self.a * (Tf - T) + self.B * k * sp) / self.c

    # ---------------- estimator ----------------
    def _ekf(self, u, y_ca, y_t):
        x = self.x
        n = 2
        c0, t0 = self._step(x[0], x[1], u, x[2], x[3], n)
        F = np.eye(4)
        eps = (1e-4, 1e-2, 1e-4, 1e-2)
        for j in range(4):
            xp = list(x)
            xp[j] += eps[j]
            cj, tj = self._step(xp[0], xp[1], u, xp[2], xp[3], n)
            F[0, j] = (cj - c0) / eps[j]
            F[1, j] = (tj - t0) / eps[j]
        xpred = np.array([c0, t0, x[2], x[3]])
        Pp = F @ self.Pm @ F.T + self.Q
        S = Pp[:2, :2] + self.R
        K = Pp[:, :2] @ np.linalg.inv(S)
        innov = np.array([y_ca - c0, y_t - t0])
        xn = xpred + K @ innov
        xn[2] = min(max(xn[2], 0.90), 1.10)
        xn[3] = min(max(xn[3], 335.0), 365.0)
        self.x = [float(v) for v in xn]
        IKH = np.eye(4)
        IKH[:, :2] -= K
        self.Pm = IKH @ Pp @ IKH.T + K @ self.R @ K.T

    # ---------------- controller ----------------
    def _resid(self, useq, x, sp, uss, u_last):
        Ca, T, Caf, Tf = x
        H, P = self.H, self.P
        sl = math.sqrt(self.p['lam'])
        wt = math.sqrt(self.p['w_t'])
        tmax = self.p['t_max']
        r = []
        up = u_last
        for j in range(P):
            u = useq[j] if j < H else uss
            Ca, T = self._step(Ca, T, u, Caf, Tf, 1)
            r.append((Ca - sp) * 100.0)
            r.append(wt * (T - tmax) if T > tmax else 0.0)
            if j < H:
                r.append(sl * (u - up))
                up = u
        return np.array(r)

    def _mpc(self, x, sp, u_last):
        H = self.H
        uss = min(max(self._u_ss(sp, x[2], x[3]), U_MIN), U_MAX)
        if self.useq is None:
            u = np.full(H, uss)
        else:
            u = np.append(self.useq[1:], uss)
        r = self._resid(u, x, sp, uss, u_last)
        cost = float(r @ r)
        du = 0.05
        for _ in range(int(self.p['iters'])):
            J = np.empty((len(r), H))
            for j in range(H):
                up = u.copy()
                d = du if u[j] + du <= U_MAX else -du
                up[j] += d
                J[:, j] = (self._resid(up, x, sp, uss, u_last) - r) / d
            g = J.T @ r
            free = ~(((u <= U_MIN + 1e-9) & (g > 0)) | ((u >= U_MAX - 1e-9) & (g < 0)))
            if not free.any():
                break
            Jf = J[:, free]
            A = Jf.T @ Jf
            A[np.diag_indices_from(A)] += 1e-6 + 1e-5 * np.trace(A) / A.shape[0]
            try:
                s = -np.linalg.solve(A, Jf.T @ r)
            except Exception:
                break
            improved = False
            step = 1.0
            for _ls in range(4):
                un = u.copy()
                un[free] = np.clip(u[free] + step * s, U_MIN, U_MAX)
                rn = self._resid(un, x, sp, uss, u_last)
                cn = float(rn @ rn)
                if cn < cost:
                    improved = True
                    break
                step *= 0.4
            if not improved:
                break
            done = cost - cn < 1e-4 * (1.0 + cost)
            u, r, cost = un, rn, cn
            if done:
                break
        self.useq = u
        return float(u[0])

    def act(self, obs):
        try:
            y_ca = float(obs['Ca'])
            y_t = float(obs['T'])
            sp = float(obs['Ca_sp'])
            if not (math.isfinite(y_ca) and math.isfinite(y_t) and math.isfinite(sp)):
                raise ValueError
            if self.x is None:
                p = self.p
                self.x = [y_ca, y_t, p['Caf0'], p['Tf0']]
                self.Pm = np.diag([p['r_ca'] ** 2, p['r_t'] ** 2, p['p0_caf'] ** 2, p['p0_tf'] ** 2])
            else:
                self._ekf(self.u_prev, y_ca, y_t)
            u_last = self.u_prev if self.u_prev is not None else 298.5
            u = self._mpc(self.x, sp, u_last)
            if not math.isfinite(u):
                u = 298.5
        except Exception:
            u = self.u_prev if self.u_prev is not None else 298.5
            self.useq = None
        u = min(max(u, U_MIN), U_MAX)
        self.u_prev = u
        return u
