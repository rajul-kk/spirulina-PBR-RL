"""EKF (augmented with the two unmeasured feed disturbances) + nonlinear MPC for the CSTR.
Model equations from pcgym cstr_ode; minutes as time unit. Scalar-math implementation."""
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


def _step(ca, T, Tc, Ti, Caf, nsub=2):
    """RK4 over one sample; plain floats."""
    h = DT / nsub
    exp = math.exp
    c0 = Ti + UAC * Tc
    c1 = 1.0 + UAC
    for _ in range(nsub):
        r = K0 * exp(-EA / T) * ca
        a1 = Caf - ca - r
        b1 = c0 - c1 * T + DH * r
        x = ca + .5 * h * a1; y = T + .5 * h * b1
        r = K0 * exp(-EA / y) * x
        a2 = Caf - x - r
        b2 = c0 - c1 * y + DH * r
        x = ca + .5 * h * a2; y = T + .5 * h * b2
        r = K0 * exp(-EA / y) * x
        a3 = Caf - x - r
        b3 = c0 - c1 * y + DH * r
        x = ca + h * a3; y = T + h * b3
        r = K0 * exp(-EA / y) * x
        a4 = Caf - x - r
        b4 = c0 - c1 * y + DH * r
        ca += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        T += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
    return ca, T


def _box_qp(G, g, lo, hi):
    """min 0.5 d.G.d + g.d subject to lo <= d <= hi, G SPD and small. Primal active set."""
    n = len(g)
    d = np.zeros(n)
    fixed = np.zeros(n, dtype=bool)
    for _ in range(3 * n):
        free = ~fixed
        if free.any():
            rhs = -(g[free] + G[np.ix_(free, fixed)] @ d[fixed])
            df = np.linalg.solve(G[np.ix_(free, free)], rhs)
            lof, hif = lo[free], hi[free]
            viol = (df < lof - 1e-12) | (df > hif + 1e-12)
            if viol.any():
                # move as far as possible along the segment, fix the blocking variable
                cur = d[free]
                step = df - cur
                t = np.full(len(df), np.inf)
                up = viol & (step > 0)
                dn = viol & (step < 0)
                t[up] = (hif[up] - cur[up]) / step[up]
                t[dn] = (lof[dn] - cur[dn]) / step[dn]
                j = int(np.argmin(t))
                tmin = max(0.0, min(1.0, float(t[j])))
                new = cur + tmin * step
                new[j] = hif[j] if step[j] > 0 else lof[j]
                idx = np.flatnonzero(free)
                d[idx] = np.minimum(np.maximum(new, lof), hif)
                fixed[idx[j]] = True
                continue
            d[free] = df
        # all free variables optimal; check multipliers of the fixed ones
        grad = G @ d + g
        rel = fixed & (((d <= lo + 1e-12) & (grad < -1e-12)) | ((d >= hi - 1e-12) & (grad > 1e-12)))
        if not rel.any():
            break
        cand = np.flatnonzero(rel)
        fixed[cand[int(np.argmax(np.abs(grad[cand])))]] = False
    return d


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
        self.bmap = [i for i, b in enumerate(blocks) for _ in range(b)]
        self.nb1 = sum(1 for b in blocks if b == 1)
        self.v = np.full(self.M, 299.5)    # decision vector (block values), warm start
        self.R = np.diag([p["r_ca"] ** 2, p["r_T"] ** 2])

    # ---------------- estimator ----------------
    def _measure(self, z):
        p = self.p
        if self.x is None:
            self.x = np.array([z[0], z[1], 350.0, 1.0])
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
        ca, T, Ti, Caf = (float(v) for v in self.x)
        c1, T1 = _step(ca, T, u, Ti, Caf, 4)
        F = np.eye(4)
        for i, e in enumerate((1e-6, 1e-3, 1e-3, 1e-6)):
            a = [ca, T, Ti, Caf]
            a[i] += e
            c2, T2 = _step(a[0], a[1], u, a[2], a[3], 4)
            F[0, i] = (c2 - c1) / e
            F[1, i] = (T2 - T1) / e
        self.x = np.array([c1, T1, Ti, Caf])
        p = self.p
        Q = np.diag([p["q_ca"] ** 2, p["q_T"] ** 2, 0.0, 0.0])
        kn = self.k + 1                    # index of the disturbance acting over the next interval
        if p["jump_lo"] <= kn < p["jump_hi"]:
            Q[2, 2] = p["q_ti"] ** 2
            Q[3, 3] = p["q_caf"] ** 2
        self.P = F @ self.P @ F.T + Q

    # ---------------- NMPC ----------------
    def _lin_rollout(self, ca, T, v, Ti, Caf, H, M, jac=True):
        """Nominal Ca trajectory (H,) and, if jac, d Ca_j / d v_m (H, M)."""
        ns = self.p["nsub"]
        bmap = self.bmap
        out = np.empty(H)
        J = np.zeros((H, M)) if jac else None
        Sc = np.zeros(M)
        ST = np.zeros(M)
        e1, e2, e3 = 1e-6, 1e-3, 1e-3
        for j in range(H):
            m = bmap[j]
            u = v[m]
            cn, Tn = _step(ca, T, u, Ti, Caf, ns)
            if jac:
                ca_, Ta_ = _step(ca + e1, T, u, Ti, Caf, ns)
                cb_, Tb_ = _step(ca, T + e2, u, Ti, Caf, ns)
                cu_, Tu_ = _step(ca, T, u + e3, Ti, Caf, ns)
                a11 = (ca_ - cn) / e1; a21 = (Ta_ - Tn) / e1
                a12 = (cb_ - cn) / e2; a22 = (Tb_ - Tn) / e2
                Sc, ST = a11 * Sc + a12 * ST, a21 * Sc + a22 * ST
                Sc[m] += (cu_ - cn) / e3
                ST[m] += (Tu_ - Tn) / e3
                J[j] = Sc
            ca, T = cn, Tn
            out[j] = ca
        return out, J

    def _mpc(self, ca0, T0, Ti, Caf, sp):
        H = min(self.H, NSTEP - self.k)
        M = self.bmap[H - 1] + 1
        v = self.v[:M].copy()
        mu = self.p["mu"]
        best_v, best_c = v.copy(), np.inf
        iters = self.p["gn_iters"]
        for it in range(iters + 1):
            last = it == iters
            out, J = self._lin_rollout(ca0, T0, [float(a) for a in v], Ti, Caf, H, M, jac=not last)
            r = out - sp
            c = float(r @ r)
            if c < best_c:
                improved = best_c - c
                best_c, best_v = c, v.copy()
                if it > 0 and improved < 1e-12:
                    break
            elif it > 0:
                v = 0.5 * (v + best_v)     # GN step overshot: back off towards the best point
                continue
            if last:
                break
            G = J.T @ J + mu * np.eye(M)
            g = J.T @ r
            v = np.minimum(np.maximum(v + _box_qp(G, g, ULO - v, UHI - v), ULO), UHI)
        self.v[:M] = best_v
        return float(best_v[0])

    def act(self, obs):
        z = np.array([obs["Ca"], obs["T"]])
        self._measure(z)
        if self.p["oracle"] and "Ti_true" in obs:
            self.x = np.array([obs["Ca"], obs["T"], obs["Ti_true"], obs["Caf_true"]])
        ca, T, Ti, Caf = (float(a) for a in self.x)
        sp = obs["Ca_sp"] + self.p["hedge"] * (0.88 - obs["Ca_sp"])
        u = self._mpc(ca, T, Ti, Caf, sp)
        u = min(max(u, ULO), UHI)
        self._predict(u)
        self.v[:self.nb1 - 1] = self.v[1:self.nb1]   # warm start: shift the single-step blocks
        self.k += 1
        return u
