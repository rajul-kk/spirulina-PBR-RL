"""EKF (augmented with the two unmeasured feed disturbances) + nonlinear MPC for the CSTR.
Model equations from pcgym cstr_ode; minutes as time unit."""
import math
import numpy as np

K0, EA = 7.2e10, 8750.0
DH = 5e4 / (1000 * 0.239)             # K per (mol/L)
UAC = 5e4 / (1000 * 0.239 * 100)      # 1/min
DT = 26.0 / 120
NSTEP = 120
ULO, UHI = 295.0, 302.0

DEFAULTS = dict(
    H=24, blocks=(1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 8), gn_iters=3, mu=1e-9, nsub=2,
    q_ti=0.17, q_caf=0.0023, q_ca=1e-5, q_T=1e-3, r_ca=0.002, r_T=0.2,
    p0_ti=0.87, p0_caf=0.0115, jump_lo=20, jump_hi=100, oracle=False, hedge=0.0,
)


def _f(ca, T, Tc, Ti, Caf):
    r = K0 * np.exp(-EA / T) * ca
    return Caf - ca - r, (Ti - T) + DH * r + UAC * (Tc - T)


def _step(ca, T, Tc, Ti, Caf, nsub=2):
    h = DT / nsub
    for _ in range(nsub):
        a1, b1 = _f(ca, T, Tc, Ti, Caf)
        a2, b2 = _f(ca + .5 * h * a1, T + .5 * h * b1, Tc, Ti, Caf)
        a3, b3 = _f(ca + .5 * h * a2, T + .5 * h * b2, Tc, Ti, Caf)
        a4, b4 = _f(ca + h * a3, T + h * b3, Tc, Ti, Caf)
        ca = ca + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        T = T + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
    return ca, T


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.k = 0
        self.x = None
        self.P = None
        self.H = int(p["H"])
        blocks = list(p["blocks"])
        assert sum(blocks) == self.H
        self.M = len(blocks)
        self.bmap = np.repeat(np.arange(self.M), blocks)
        self.v = None                      # decision vector (block values)
        self.R = np.diag([p["r_ca"] ** 2, p["r_T"] ** 2])
        self.eps_fd = np.array([1e-6, 1e-3, 1e-3, 1e-6])

    # ---------------- estimator ----------------
    def _measure(self, z):
        if self.x is None:
            self.x = np.array([z[0], z[1], 350.0, 1.0])
            p = self.p
            self.P = np.diag([p["r_ca"] ** 2, p["r_T"] ** 2, p["p0_ti"] ** 2, p["p0_caf"] ** 2])
            return
        P = self.P
        S = P[:2, :2] + self.R
        K = P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x[:2])
        IKH = np.eye(4)
        IKH[:, :2] -= K
        self.P = IKH @ P @ IKH.T + K @ self.R @ K.T
        self.x[2] = min(max(self.x[2], 348.5), 351.5)
        self.x[3] = min(max(self.x[3], 0.98), 1.02)

    def _predict(self, u):
        x = self.x
        X = x[None, :] + np.vstack([np.zeros(4), np.diag(self.eps_fd)])
        ca, T = _step(X[:, 0], X[:, 1], u, X[:, 2], X[:, 3], 4)
        F = np.eye(4)
        F[0, :] = (ca[1:] - ca[0]) / self.eps_fd
        F[1, :] = (T[1:] - T[0]) / self.eps_fd
        self.x = np.array([ca[0], T[0], x[2], x[3]])
        p = self.p
        Q = np.diag([p["q_ca"] ** 2, p["q_T"] ** 2, 0.0, 0.0])
        # a feed shift can only happen at samples jump_lo..jump_hi-1 (the disturbance used
        # during the *next* interval is the one that matters for the next prediction)
        kn = self.k + 1
        if p["jump_lo"] <= kn < p["jump_hi"]:
            Q[2, 2] = p["q_ti"] ** 2
            Q[3, 3] = p["q_caf"] ** 2
        self.P = F @ self.P @ F.T + Q

    # ---------------- NMPC ----------------
    def _rollout(self, ca0, T0, V, Ti, Caf, H):
        """V: (n, M) block values -> Ca trajectories (n, H)."""
        U = V[:, self.bmap[:H]]
        n = V.shape[0]
        ca = np.full(n, ca0)
        T = np.full(n, T0)
        out = np.empty((n, H))
        ns = self.p["nsub"]
        for j in range(H):
            ca, T = _step(ca, T, U[:, j], Ti, Caf, ns)
            out[:, j] = ca
        return out

    def _mpc(self, ca0, T0, Ti, Caf, sp):
        H = min(self.H, NSTEP - self.k)
        M = int(self.bmap[H - 1]) + 1
        if self.v is None:
            self.v = np.full(self.M, 299.5)
        v = self.v[:M].copy()
        eps = 1e-3
        mu = self.p["mu"]
        best_v, best_c = None, np.inf
        for it in range(self.p["gn_iters"] + 1):
            V = np.vstack([v[None, :], v[None, :] + eps * np.eye(M)])
            out = self._rollout(ca0, T0, V, Ti, Caf, H)
            r = out[0] - sp
            c = float(r @ r)
            if c < best_c:
                best_c, best_v = c, v.copy()
            elif it > 0:
                # GN step overshot: back off halfway towards the best point and retry
                v = 0.5 * (v + best_v)
                continue
            if it == self.p["gn_iters"]:
                break
            J = (out[1:] - out[0]).T / eps          # (H, M)
            G = J.T @ J + mu * np.eye(M)
            g = J.T @ r
            lo, hi = ULO - v, UHI - v
            d = np.zeros(M)
            for _ in range(40):                     # projected Gauss-Seidel on the box QP
                dmax = 0.0
                for i in range(M):
                    new = d[i] - (g[i] + G[i] @ d) / G[i, i]
                    new = min(max(new, lo[i]), hi[i])
                    dmax = max(dmax, abs(new - d[i]))
                    d[i] = new
                if dmax < 1e-4:
                    break
            v = v + d
        self.v[:M] = best_v
        return float(best_v[0])

    def act(self, obs):
        z = np.array([obs["Ca"], obs["T"]])
        self._measure(z)
        if self.p["oracle"] and "Ti_true" in obs:
            self.x = np.array([obs["Ca"], obs["T"], obs["Ti_true"], obs["Caf_true"]])
        ca, T, Ti, Caf = self.x
        sp = obs["Ca_sp"] + self.p["hedge"] * (0.88 - obs["Ca_sp"])
        u = self._mpc(ca, T, Ti, Caf, sp)
        u = min(max(u, ULO), UHI)
        self._predict(u)
        # warm start for next sample: shift the single-step blocks
        v = self.v
        nb1 = sum(1 for b in self.p["blocks"] if b == 1)
        v[:nb1 - 1] = v[1:nb1]
        self.k += 1
        return u
