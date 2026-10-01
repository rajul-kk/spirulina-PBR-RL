"""Ca controller for the jacketed CSTR: extended Kalman filter + nonlinear model-predictive control.

Model (identified from pilot batches b002-b009; literature values fit to within noise), time in minutes:
    dCa/dt = a (Caf - Ca) - k(T) Ca
    dT/dt  = a (Tf - T) + beta k(T) Ca + alpha (Tc - T),      k(T) = k0 exp(-ER / T)
Unmeasured feed disturbances Caf, Tf are estimated as extra (random-walk) states.
"""
import math
import numpy as np

DEFAULTS = dict(
    a=1.0, k0=7.2e10, ER=8750.0, beta=209.2, alpha=2.092,
    dt=26.0 / 120.0,
    u_min=295.0, u_max=302.0,
    r_ca=0.0018, r_t=0.19,            # measurement noise std
    q_ca=2e-4, q_t=0.02,              # process noise std per sample
    q_caf=0.0009, q_tf=0.07,          # disturbance random-walk std per sample
    p0_caf=0.015, p0_tf=1.5,          # initial disturbance uncertainty
    caf0=1.005, tf0=350.3,
    blocks=(1, 1, 1, 2, 3, 8),        # move blocking; horizon = sum (samples)
    w_du=0.05,                        # move penalty weight (per K, error in units of 0.01 mol/L)
    iters=3,
    t_safe=331.0,                     # full cooling above this reactor temperature
)


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.a, self.k0, self.ER, self.beta, self.alpha = p["a"], p["k0"], p["ER"], p["beta"], p["alpha"]
        self.dt = p["dt"]
        self.x = None
        self.P = None
        self.u_prev = None
        self.Q = np.diag([p["q_ca"] ** 2, p["q_t"] ** 2, p["q_caf"] ** 2, p["q_tf"] ** 2])
        self.R = np.diag([p["r_ca"] ** 2, p["r_t"] ** 2])
        self.blocks = list(p["blocks"])
        self.useq = None

    # ---- model -------------------------------------------------------------------------------
    def _step(self, ca, t, caf, tf, tc, nsub=1):
        a, k0, ER, beta, alpha = self.a, self.k0, self.ER, self.beta, self.alpha
        h = self.dt / nsub
        exp = math.exp
        for _ in range(nsub):
            k = k0 * exp(-ER / t)
            c1 = a * (caf - ca) - k * ca
            t1 = a * (tf - t) + beta * k * ca + alpha * (tc - t)
            cb = ca + 0.5 * h * c1; tb = t + 0.5 * h * t1
            k = k0 * exp(-ER / tb)
            c2 = a * (caf - cb) - k * cb
            t2 = a * (tf - tb) + beta * k * cb + alpha * (tc - tb)
            cb = ca + 0.5 * h * c2; tb = t + 0.5 * h * t2
            k = k0 * exp(-ER / tb)
            c3 = a * (caf - cb) - k * cb
            t3 = a * (tf - tb) + beta * k * cb + alpha * (tc - tb)
            cb = ca + h * c3; tb = t + h * t3
            k = k0 * exp(-ER / tb)
            c4 = a * (caf - cb) - k * cb
            t4 = a * (tf - tb) + beta * k * cb + alpha * (tc - tb)
            ca = ca + h / 6.0 * (c1 + 2 * c2 + 2 * c3 + c4)
            t = t + h / 6.0 * (t1 + 2 * t2 + 2 * t3 + t4)
            if t > 400.0:
                t = 400.0
            elif t < 250.0:
                t = 250.0
            if ca < 0.0:
                ca = 0.0
        return ca, t

    # ---- estimator ---------------------------------------------------------------------------
    def _predict(self, u):
        x = self.x
        ca, t = self._step(x[0], x[1], x[2], x[3], u, 2)
        F = np.eye(4)
        eps = (1e-5, 1e-3, 1e-5, 1e-3)
        for j in range(4):
            xp = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
            xp[j] += eps[j]
            cj, tj = self._step(xp[0], xp[1], xp[2], xp[3], u, 2)
            F[0, j] = (cj - ca) / eps[j]
            F[1, j] = (tj - t) / eps[j]
        self.x = np.array([ca, t, x[2], x[3]])
        self.P = F @ self.P @ F.T + self.Q

    def _update(self, ca_m, t_m):
        P = self.P
        S = P[:2, :2] + self.R
        K = P[:, :2] @ np.linalg.inv(S)
        innov = np.array([ca_m - self.x[0], t_m - self.x[1]])
        self.x = self.x + K @ innov
        P = P - K @ P[:2, :]
        self.P = 0.5 * (P + P.T)
        self.x[2] = min(max(self.x[2], 0.9), 1.1)
        self.x[3] = min(max(self.x[3], 340.0), 360.0)

    # ---- controller --------------------------------------------------------------------------
    def _resid(self, v, x, sp):
        ca, t, caf, tf = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        out = []
        up = self.u_prev
        wdu = self.p["w_du"]
        step = self._step
        for b, n in enumerate(self.blocks):
            u = float(v[b])
            for _ in range(n):
                ca, t = step(ca, t, caf, tf, u)
                out.append((ca - sp) * 100.0)
            if wdu > 0.0:
                out.append(wdu * (u - up))
                up = u
        return np.array(out)

    def _mpc(self, x, sp):
        lo, hi = self.p["u_min"], self.p["u_max"]
        nb = len(self.blocks)
        if self.useq is None:
            v = np.full(nb, self.u_prev)
        else:
            v = np.append(self.useq[1:], self.useq[-1])
        v = np.clip(v, lo, hi)
        r = self._resid(v, x, sp)
        f = float(r @ r)
        lam = 1e-3
        for _ in range(self.p["iters"]):
            J = np.empty((len(r), nb))
            for j in range(nb):
                d = 0.05 if v[j] + 0.05 <= hi else -0.05
                vp = v.copy()
                vp[j] += d
                J[:, j] = (self._resid(vp, x, sp) - r) / d
            g = J.T @ r
            A = J.T @ J
            improved = False
            for _try in range(4):
                free = ~(((v <= lo + 1e-9) & (g > 0)) | ((v >= hi - 1e-9) & (g < 0)))
                step = np.zeros(nb)
                if free.any():
                    Af = A[np.ix_(free, free)] + lam * np.diag(np.diag(A)[free] + 1e-9)
                    step[free] = -np.linalg.solve(Af, g[free])
                vn = np.clip(v + step, lo, hi)
                rn = self._resid(vn, x, sp)
                fn = float(rn @ rn)
                if fn < f:
                    v, r, f = vn, rn, fn
                    lam = max(lam * 0.3, 1e-6)
                    improved = True
                    break
                lam *= 10.0
            if not improved:
                break
        self.useq = v
        return float(v[0])

    def act(self, obs):
        p = self.p
        ca_m = float(obs["Ca"])
        t_m = float(obs["T"])
        sp = float(obs["Ca_sp"])
        if self.x is None:
            self.x = np.array([ca_m, t_m, p["caf0"], p["tf0"]])
            self.P = np.diag([p["r_ca"] ** 2, p["r_t"] ** 2, p["p0_caf"] ** 2, p["p0_tf"] ** 2])
            self.u_prev = 298.5
        else:
            self._predict(self.u_prev)
            self._update(ca_m, t_m)
        try:
            u = self._mpc(self.x.copy(), sp)
        except Exception:
            u = self.u_prev
        if not (u == u):
            u = self.u_prev
        if t_m > p["t_safe"] or self.x[1] > p["t_safe"]:
            u = p["u_min"]
        u = min(max(u, p["u_min"]), p["u_max"])
        self.u_prev = u
        return u
