"""Controller for the jacketed CSTR: EKF disturbance estimator + nonlinear MPC.

Model (from the plant design model, time in minutes):
    dCa/dt = (Caf - Ca) - k(T) Ca
    dT/dt  = (Ti - T) + DH k(T) Ca + UA (Tc - T),   k(T) = K0 exp(-ER / T)
The feed temperature Ti and feed concentration Caf are unmeasured step disturbances; they are
estimated together with Ca and T by an extended Kalman filter whose disturbance covariance is
re-opened when the innovations say a step has happened. The jacket temperature is then chosen
by a short-horizon nonlinear MPC (bounded Gauss-Newton) on the same model.
"""
import math

import numpy as np

K0 = 7.2e10
ER = 8750.0
DH = 5e4 / (1000.0 * 0.239)             # K per (mol/L)
UA = 5e4 / (1000.0 * 0.239 * 100.0)     # 1/min
DT = 26.0 / 120.0                       # min per sample
U_LO, U_HI = 295.0, 302.0
TI_LO, TI_HI = 348.5, 351.5
CAF_LO, CAF_HI = 0.98, 1.02

DEFAULTS = {
    "H": 16,            # prediction horizon, samples
    "M": 8,             # free moves
    "iters": 3,         # Gauss-Newton iterations per sample
    "rho": 1e-3,        # move penalty, cost units per K^2
    "lam": 1e-4,        # Gauss-Newton damping
    "t_soft": 331.0,    # soft ceiling on predicted T, K
    "w_t": 3.0,         # weight on T excess
    "q_ti": 0.14,       # random-walk sd of Ti per sample, K
    "q_caf": 0.0018,    # random-walk sd of Caf per sample, mol/L
    "q_ca": 1e-5,       # model noise sd on Ca
    "q_t": 1e-3,        # model noise sd on T
    "r_ca": 0.002,
    "r_t": 0.2,
    "jump": 0,          # 1: innovation-triggered covariance reset
    "jump_thr": 9.0,
    "jump_n": 2,
    "nsub": 1,
    "oracle": 0,
}


def _f(ca, T, Tc, Ti, Caf):
    r = K0 * np.exp(-ER / T) * ca
    return (Caf - ca) - r, (Ti - T) + DH * r + UA * (Tc - T)


def _step(ca, T, Tc, Ti, Caf, nsub=1):
    """One sample of the plant (RK4); works on scalars or arrays."""
    h = DT / nsub
    for _ in range(nsub):
        a1, b1 = _f(ca, T, Tc, Ti, Caf)
        a2, b2 = _f(ca + 0.5 * h * a1, T + 0.5 * h * b1, Tc, Ti, Caf)
        a3, b3 = _f(ca + 0.5 * h * a2, T + 0.5 * h * b2, Tc, Ti, Caf)
        a4, b4 = _f(ca + h * a3, T + h * b3, Tc, Ti, Caf)
        ca = ca + h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        T = T + h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
    return ca, T


def steady_tc(sp, Ti, Caf):
    """Jacket temperature that holds Ca = sp at steady state."""
    sp = min(max(sp, 1e-3), Caf - 1e-4)
    k = (Caf - sp) / sp
    T = ER / math.log(K0 / k)
    return T - ((Ti - T) + DH * k * sp) / UA, T


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.H, self.M = int(p["H"]), int(p["M"])
        self.x = None
        self.P = None
        self.u_prev = None
        self.u_plan = None
        self.truth = None          # only used by my own simulator (oracle studies)
        self.nis_hist = []
        self.Q = np.diag([p["q_ca"] ** 2, p["q_t"] ** 2, p["q_ti"] ** 2, p["q_caf"] ** 2])
        self.R = np.diag([p["r_ca"] ** 2, p["r_t"] ** 2])
        self.P0d = np.array([0.87 ** 2, 0.0115 ** 2])

    # ------------------------------------------------------------------ estimator
    def _predict(self, x, u):
        d = np.array([1e-5, 1e-2, 1e-2, 1e-5])
        ca = np.full(5, x[0]); T = np.full(5, x[1]); Ti = np.full(5, x[2]); Caf = np.full(5, x[3])
        ca[1] += d[0]; T[2] += d[1]; Ti[3] += d[2]; Caf[4] += d[3]
        can, Tn = _step(ca, T, u, Ti, Caf, int(self.p["nsub"]))
        F = np.eye(4)
        F[0, :] = (can[1:] - can[0]) / d
        F[1, :] = (Tn[1:] - Tn[0]) / d
        return np.array([can[0], Tn[0], x[2], x[3]]), F

    def _estimate(self, y):
        p = self.p
        if self.x is None:
            self.x = np.array([y[0], y[1], 350.0, 1.0])
            self.P = np.diag([p["r_ca"] ** 2, p["r_t"] ** 2, self.P0d[0], self.P0d[1]])
            return
        xp, F = self._predict(self.x, self.u_prev)
        Pp = F @ self.P @ F.T + self.Q
        Hm = np.zeros((2, 4)); Hm[0, 0] = 1.0; Hm[1, 1] = 1.0
        inn = y - xp[:2]
        S = Pp[:2, :2] + self.R
        Si = np.linalg.inv(S)
        nis = float(inn @ Si @ inn)
        self.nis_hist.append(nis)
        if p["jump"] and len(self.nis_hist) >= p["jump_n"]:
            n = int(p["jump_n"])
            if sum(self.nis_hist[-n:]) > p["jump_thr"] * n / 2.0 + 2.0 * n:
                # a step in the feed is likely: re-open the disturbance covariance
                Pp[2, 2] = max(Pp[2, 2], self.P0d[0] * p.get("jump_scale", 1.0))
                Pp[3, 3] = max(Pp[3, 3], self.P0d[1] * p.get("jump_scale", 1.0))
                S = Pp[:2, :2] + self.R
                Si = np.linalg.inv(S)
                self.nis_hist = []
        K = Pp[:, :2] @ Si
        x = xp + K @ inn
        IKH = np.eye(4) - K @ Hm
        self.P = IKH @ Pp @ IKH.T + K @ self.R @ K.T
        x[2] = min(max(x[2], TI_LO), TI_HI)
        x[3] = min(max(x[3], CAF_LO), CAF_HI)
        self.x = x

    # ------------------------------------------------------------------ MPC
    def _rollout(self, U, ca0, T0, Ti, Caf):
        n, H = U.shape
        ca = np.full(n, ca0); T = np.full(n, T0)
        CA = np.empty((n, H)); TT = np.empty((n, H))
        nsub = int(self.p["nsub"])
        for j in range(H):
            ca, T = _step(ca, T, U[:, j], Ti, Caf, nsub)
            CA[:, j] = ca; TT[:, j] = T
        return CA, TT

    def _resid(self, Um, CA, TT, sp, u_last):
        """Residual vectors (n, nres) for move sets Um (n, M)."""
        p = self.p
        r1 = (CA - sp) / 0.01
        r2 = p["w_t"] * np.maximum(TT - p["t_soft"], 0.0)
        du = np.diff(np.concatenate([np.full((Um.shape[0], 1), u_last), Um], axis=1), axis=1)
        r3 = math.sqrt(p["rho"]) * du
        return np.concatenate([r1, r2, r3], axis=1)

    def _full(self, Um, u_ss):
        n = Um.shape[0]
        return np.concatenate([Um, np.full((n, self.H - self.M), u_ss)], axis=1)

    def _mpc(self, ca0, T0, Ti, Caf, sp):
        p = self.p
        M = self.M
        u_ss, _ = steady_tc(sp, Ti, Caf)
        u_ss = min(max(u_ss, U_LO), U_HI)
        u_last = self.u_prev if self.u_prev is not None else u_ss
        if self.u_plan is None:
            u = np.full(M, u_ss)
        else:
            u = np.concatenate([self.u_plan[1:], [u_ss]])
        u = np.clip(u, U_LO, U_HI)
        eps = 1e-3
        for _ in range(int(p["iters"])):
            s = np.where(u > U_HI - 2 * eps, -1.0, 1.0)
            Um = np.tile(u, (M + 1, 1))
            for i in range(M):
                Um[i + 1, i] += s[i] * eps
            CA, TT = self._rollout(self._full(Um, u_ss), ca0, T0, Ti, Caf)
            Rs = self._resid(Um, CA, TT, sp, u_last)
            r0 = Rs[0]
            J = ((Rs[1:] - r0) / (s[:, None] * eps)).T          # (nres, M)
            A = J.T @ J + p["lam"] * np.eye(M)
            g = J.T @ r0
            lo = U_LO - u; hi = U_HI - u
            du = np.zeros(M)
            for _sweep in range(40):
                mx = 0.0
                for i in range(M):
                    v = -(g[i] + A[i] @ du - A[i, i] * du[i]) / A[i, i]
                    v = min(max(v, lo[i]), hi[i])
                    mx = max(mx, abs(v - du[i]))
                    du[i] = v
                if mx < 1e-5:
                    break
            alphas = np.array([1.0, 0.5, 0.25, 0.1, 0.0])
            Uc = np.clip(u[None, :] + alphas[:, None] * du[None, :], U_LO, U_HI)
            CA, TT = self._rollout(self._full(Uc, u_ss), ca0, T0, Ti, Caf)
            cost = np.sum(self._resid(Uc, CA, TT, sp, u_last) ** 2, axis=1)
            cost = np.where(np.isfinite(cost), cost, np.inf)
            b = int(np.argmin(cost))
            u = Uc[b]
            if alphas[b] == 0.0 or np.max(np.abs(du)) * alphas[b] < 1e-3:
                break
        self.u_plan = u
        return float(u[0])

    # ------------------------------------------------------------------ interface
    def act(self, obs):
        sp = float(obs["Ca_sp"])
        y = np.array([float(obs["Ca"]), float(obs["T"])])
        try:
            self._estimate(y)
            if self.p["oracle"] and self.truth is not None:
                ca0, T0, Ti, Caf = self.truth
            else:
                ca0, T0, Ti, Caf = self.x
            if T0 > 333.0:
                u = U_LO
                self.u_plan = None
            else:
                u = self._mpc(ca0, T0, Ti, Caf, sp)
            if not math.isfinite(u):
                raise ValueError("non-finite")
        except Exception:
            # fall back on a simple steady-state jacket temperature
            try:
                u, _ = steady_tc(sp, 350.0, 1.0)
            except Exception:
                u = 299.0
            self.x = None
            self.u_plan = None
        u = min(max(u, U_LO), U_HI)
        self.u_prev = u
        return u
