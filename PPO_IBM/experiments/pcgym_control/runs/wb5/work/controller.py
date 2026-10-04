"""CSTR concentration controller: EKF (Ca, T, feed temperature Ti, feed concentration Caf)
+ nonlinear MPC on the first-principles model (single shooting, Gauss-Newton with box limits).
Allowed imports only: numpy, math, collections."""
import math
import numpy as np

DT = 26.0 / 120.0
LO, HI = 295.0, 302.0

DEFAULTS = {
    "H": 14,                 # prediction horizon (samples)
    "blocks": (1, 1, 1, 1, 2, 2, 3, 3),   # move blocking of the input over the horizon
    "w_du": 0.0,             # penalty on input moves (per K, in error units of 0.01 mol/L)
    "w_u_end": 0.0,
    "T_soft": 333.0,         # soft temperature limit in the prediction (K)
    "w_T": 50.0,
    "gn_iter": 6,
    "nsub": 2,               # RK4 substeps per sample in the predictor
    "q_ca": 1e-7, "q_T": 1e-3, "q_Ti": 0.02, "q_Caf": 4e-6,   # EKF process noise variances / sample
    "p0_Ti": 1.0, "p0_Caf": 1.5e-4,
    "r_ca": 0.002 ** 2, "r_T": 0.2 ** 2,
}


def _rhs(x, Tc, Ti, Caf):
    ca, T = x[..., 0], x[..., 1]
    T = np.minimum(T, 420.0)
    rA = 7.2e10 * np.exp(-8750.0 / T) * ca
    dca = (Caf - ca) - rA
    dT = (Ti - T) + (5e4 / 239.0) * rA + (5e4 / 23900.0) * (Tc - T)
    return np.stack([dca, dT], axis=-1)


def _step(x, Tc, Ti, Caf, nsub):
    h = DT / nsub
    for _ in range(nsub):
        k1 = _rhs(x, Tc, Ti, Caf)
        k2 = _rhs(x + 0.5 * h * k1, Tc, Ti, Caf)
        k3 = _rhs(x + 0.5 * h * k2, Tc, Ti, Caf)
        k4 = _rhs(x + h * k3, Tc, Ti, Caf)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.k = 0
        self.z = None          # [Ca, T, Ti, Caf]
        self.P = None
        self.u_prev = 299.5
        nb = len(p["blocks"])
        self.v = np.full(nb, 299.5)   # warm start of blocked input
        # expansion matrix from blocked moves to horizon
        H = p["H"]
        E = np.zeros((H, nb))
        i = 0
        for j, b in enumerate(p["blocks"]):
            for _ in range(b):
                if i < H:
                    E[i, j] = 1.0
                    i += 1
        while i < H:
            E[i, nb - 1] = 1.0
            i += 1
        self.E = E

    # ---------------- estimator ----------------
    def _f(self, z, u):
        x = _step(z[:2], u, z[2], z[3], 4)
        return np.array([x[0], x[1], z[2], z[3]])

    def _ekf(self, y):
        p = self.p
        if self.z is None:
            self.z = np.array([y[0], y[1], 350.0, 1.0])
            self.P = np.diag([p["r_ca"], p["r_T"], p["p0_Ti"], p["p0_Caf"]])
        else:
            z0 = self.z
            fz = self._f(z0, self.u_prev)
            F = np.zeros((4, 4))
            eps = np.array([1e-5, 1e-3, 1e-3, 1e-5])
            for i in range(4):
                d = np.zeros(4)
                d[i] = eps[i]
                F[:, i] = (self._f(z0 + d, self.u_prev) - self._f(z0 - d, self.u_prev)) / (2 * eps[i])
            Q = np.diag([p["q_ca"], p["q_T"], p["q_Ti"], p["q_Caf"]])
            self.z = fz
            self.P = F @ self.P @ F.T + Q
        Hm = np.zeros((2, 4))
        Hm[0, 0] = 1.0
        Hm[1, 1] = 1.0
        R = np.diag([p["r_ca"], p["r_T"]])
        S = Hm @ self.P @ Hm.T + R
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.z = self.z + K @ (np.asarray(y) - self.z[:2])
        self.P = (np.eye(4) - K @ Hm) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        # keep disturbance estimates physically sane
        self.z[2] = min(max(self.z[2], 340.0), 360.0)
        self.z[3] = min(max(self.z[3], 0.9), 1.1)

    # ---------------- MPC ----------------
    def _residuals(self, V, x0, Ti, Caf, sp):
        """V: (m, nb) candidate blocked inputs -> residual matrix (m, nres)."""
        p = self.p
        U = V @ self.E.T                       # (m, H)
        m, H = U.shape
        x = np.repeat(x0[None, :], m, axis=0)
        r_ca = np.empty((m, H))
        r_T = np.empty((m, H))
        for i in range(H):
            x = _step(x, U[:, i], Ti, Caf, p["nsub"])
            r_ca[:, i] = (x[:, 0] - sp) / 0.01
            r_T[:, i] = math.sqrt(p["w_T"]) * np.maximum(x[:, 1] - p["T_soft"], 0.0)
        res = [r_ca, r_T]
        if p["w_du"] > 0:
            du = np.diff(np.concatenate([np.full((m, 1), self.u_prev), V], axis=1), axis=1)
            res.append(math.sqrt(p["w_du"]) * du)
        return np.concatenate(res, axis=1)

    def _lin(self, v, x0, Ti, Caf, sp, h=0.01):
        nb = v.size
        sign = np.where(v + h > HI, -1.0, 1.0)
        cand = np.vstack([v[None, :], v[None, :] + h * np.diag(sign)])
        R = self._residuals(cand, x0, Ti, Caf, sp)
        r = R[0]
        J = ((R[1:] - r[None, :]) * sign[:, None]).T / h
        return r, J

    def _mpc(self, x0, Ti, Caf, sp):
        p = self.p
        v = np.clip(self.v, LO, HI)
        nb = v.size
        r, J = self._lin(v, x0, Ti, Caf, sp)
        f = r @ r
        for it in range(p["gn_iter"]):
            A = J.T @ J + 1e-4 * np.eye(nb)
            bvec = J.T @ r
            lo, hi = LO - v, HI - v
            dv = np.zeros(nb)
            for _ in range(60):          # box-constrained QP by coordinate descent
                old = dv.copy()
                for i in range(nb):
                    gi = bvec[i] + A[i] @ dv - A[i, i] * dv[i]
                    dv[i] = min(max(-gi / A[i, i], lo[i]), hi[i])
                if np.abs(dv - old).max() < 1e-4:
                    break
            alphas = np.array([1.0, 0.5, 0.25, 0.1])
            trials = np.clip(v[None, :] + alphas[:, None] * dv[None, :], LO, HI)
            Rt = self._residuals(trials, x0, Ti, Caf, sp)
            ft = np.einsum("ij,ij->i", Rt, Rt)
            j = int(np.argmin(ft))
            if ft[j] >= f - 1e-9:
                break
            v, f = trials[j], ft[j]
            if it < p["gn_iter"] - 1:
                r, J = self._lin(v, x0, Ti, Caf, sp)
        self.v = np.concatenate([v[1:], v[-1:]]) if p["blocks"][0] == 1 else v
        return float(v[0])

    def act(self, obs):
        y = (float(obs["Ca"]), float(obs["T"]))
        self._ekf(y)
        z = self.z
        u = self._mpc(z[:2].copy(), z[2], z[3], float(obs["Ca_sp"]))
        u = min(max(u, LO), HI)
        self.u_prev = u
        self.k += 1
        return u
