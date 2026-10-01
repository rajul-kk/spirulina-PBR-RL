"""Jacketed CSTR concentration controller (run bb2).

Design: extended Kalman filter on a fitted first-principles model (states Ca, T plus the two
unmeasured feed disturbances Caf, Tf) feeding a short-horizon nonlinear model-predictive
controller that minimises predicted squared Ca error subject to the jacket limits.
All model numbers come from the pilot batches (see LAB_NOTEBOOK.md).
"""
import math
import numpy as np

DEFAULTS = dict(
    # model (time unit: minutes)
    a=1.0, kref=0.11, ER=8750.0, b=209.0, c=2.09, Tref=322.0,
    Caf0=1.0, Tf0=350.0,
    dt=26.0 / 120.0,
    # filter
    sCa=0.003, sT=0.3, qCa=1e-4, qT=1e-2, qCaf=0.005, qTf=0.5,
    p0Caf=0.03, p0Tf=3.0,
    # controller
    N=12, M=4, rho=0.0, iters=3,
    umin=295.0, umax=302.0,
    Tsafe=331.5,
)


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.x = None
        self.P = None
        self.useq = None
        self.u_prev = None
        self.R = np.diag([p["sCa"] ** 2, p["sT"] ** 2])
        self.Q = np.diag([p["qCa"] ** 2, p["qT"] ** 2, p["qCaf"] ** 2, p["qTf"] ** 2])

    # ---------- model ----------
    def _k(self, T):
        p = self.p
        return p["kref"] * math.exp(-p["ER"] * (1.0 / T - 1.0 / p["Tref"]))

    def _f(self, Ca, T, Caf, Tf, u):
        p = self.p
        k = self._k(T)
        return (p["a"] * (Caf - Ca) - k * Ca,
                p["a"] * (Tf - T) + p["b"] * k * Ca + p["c"] * (u - T))

    def _step(self, Ca, T, Caf, Tf, u):
        h = self.p["dt"]
        k1 = self._f(Ca, T, Caf, Tf, u)
        k2 = self._f(Ca + 0.5 * h * k1[0], T + 0.5 * h * k1[1], Caf, Tf, u)
        k3 = self._f(Ca + 0.5 * h * k2[0], T + 0.5 * h * k2[1], Caf, Tf, u)
        k4 = self._f(Ca + h * k3[0], T + h * k3[1], Caf, Tf, u)
        Ca2 = Ca + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        T2 = T + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return Ca2, min(max(T2, 250.0), 450.0)

    def _jac(self, x, u):
        F = np.eye(4)
        eps = (1e-5, 1e-3, 1e-5, 1e-3)
        for i in range(4):
            xp = list(x)
            xm = list(x)
            xp[i] += eps[i]
            xm[i] -= eps[i]
            a = self._step(xp[0], xp[1], xp[2], xp[3], u)
            b = self._step(xm[0], xm[1], xm[2], xm[3], u)
            F[0, i] = (a[0] - b[0]) / (2 * eps[i])
            F[1, i] = (a[1] - b[1]) / (2 * eps[i])
        return F

    # ---------- estimator ----------
    def _estimate(self, y):
        p = self.p
        if self.x is None:
            self.x = np.array([y[0], y[1], p["Caf0"], p["Tf0"]])
            self.P = np.diag([p["sCa"] ** 2, p["sT"] ** 2, p["p0Caf"] ** 2, p["p0Tf"] ** 2])
            return
        F = self._jac(self.x, self.u_prev)
        Ca, T = self._step(self.x[0], self.x[1], self.x[2], self.x[3], self.u_prev)
        x = np.array([Ca, T, self.x[2], self.x[3]])
        P = F.dot(self.P).dot(F.T) + self.Q
        e = np.array([y[0] - x[0], y[1] - x[1]])
        S = P[:2, :2] + self.R
        K = P[:, :2].dot(np.linalg.inv(S))
        x = x + K.dot(e)
        P = (np.eye(4) - K.dot(np.eye(2, 4))).dot(P)
        x[0] = min(max(x[0], 0.0), 2.0)
        x[2] = min(max(x[2], 0.5), 1.5)
        x[3] = min(max(x[3], 300.0), 400.0)
        self.x = x
        self.P = 0.5 * (P + P.T)

    # ---------- control ----------
    def _uss(self, sp):
        p = self.p
        Caf, Tf = self.x[2], self.x[3]
        ks = p["a"] * max(Caf - sp, 1e-4) / sp
        Ts = 1.0 / (1.0 / p["Tref"] - math.log(ks / p["kref"]) / p["ER"])
        u = Ts - (p["a"] * (Tf - Ts) + p["b"] * ks * sp) / p["c"]
        return min(max(u, p["umin"]), p["umax"])

    def _resid(self, useq, uss, sp):
        p = self.p
        Ca, T, Caf, Tf = self.x
        N, M = p["N"], p["M"]
        r = np.empty(N + M)
        for j in range(N):
            u = useq[j] if j < M else uss
            Ca, T = self._step(Ca, T, Caf, Tf, u)
            r[j] = (Ca - sp) / 0.01
        w = math.sqrt(p["rho"])
        for j in range(M):
            r[N + j] = w * (useq[j] - uss)
        return r

    def _mpc(self, sp):
        p = self.p
        M = p["M"]
        lo, hi = p["umin"], p["umax"]
        uss = self._uss(sp)
        if self.useq is None:
            useq = np.full(M, uss)
        else:
            useq = np.append(self.useq[1:], uss)
        r = self._resid(useq, uss, sp)
        cost = r.dot(r)
        lam = 1e-3
        for _ in range(p["iters"]):
            J = np.empty((len(r), M))
            for i in range(M):
                d = 0.05 if useq[i] + 0.05 <= hi else -0.05
                up = useq.copy()
                up[i] += d
                J[:, i] = (self._resid(up, uss, sp) - r) / d
            g = J.T.dot(r)
            Hm = J.T.dot(J)
            improved = False
            for _try in range(4):
                try:
                    du = -np.linalg.solve(Hm + lam * np.eye(M), g)
                except Exception:
                    break
                un = np.clip(useq + du, lo, hi)
                rn = self._resid(un, uss, sp)
                cn = rn.dot(rn)
                if cn < cost:
                    useq, r, cost = un, rn, cn
                    lam = max(lam * 0.3, 1e-6)
                    improved = True
                    break
                lam *= 10.0
            if not improved:
                break
        self.useq = useq
        return float(useq[0])

    def act(self, obs):
        p = self.p
        try:
            y = (float(obs["Ca"]), float(obs["T"]))
            sp = float(obs["Ca_sp"])
            self._estimate(y)
            u = self._mpc(sp)
            if self.x[1] > p["Tsafe"] or y[1] > p["Tsafe"] + 1.0:
                u = p["umin"]
            if not (u == u):
                u = 298.5
        except Exception:
            u = 298.5 if self.u_prev is None else self.u_prev
        u = min(max(u, p["umin"]), p["umax"])
        self.u_prev = u
        return u
