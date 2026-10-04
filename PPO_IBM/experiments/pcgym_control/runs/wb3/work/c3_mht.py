"""Jump-aware multi-hypothesis EKF (state + the two unmeasured feed disturbances) feeding a
nonlinear MPC for the CSTR. Model equations from pcgym cstr_ode; minutes as time unit."""
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
    q_ca=1e-5, q_T=1e-3, r_ca=0.002, r_T=0.2,
    p0_ti=0.866, p0_caf=0.01155, n_hyp=16, oracle=False, hedge_c=2.0, haz_scale=1.0,
)
C_LO, C_HI = 20, 100                  # a step change can occur at sample indices 20..99
NPAIR = 1830.0                        # pairs (c1 < c2) with c2 - c1 >= 20 in 20..99


def haz_dist(k, n, c1):
    """P(feed shift at sample k | history): n shifts so far (first one at c1). 1 or 2 shifts
    per batch (equally likely a priori), each segment at least 20 samples long."""
    if k < C_LO or k >= C_HI or n >= 2:
        return 0.0
    if n == 0:
        m = max(0, 80 - k)
        surv = 0.5 * (100 - k) / 80.0 + 0.5 * (m * (m + 1) / 2.0) / NPAIR
        return (0.5 / 80.0 + 0.5 * m / NPAIR) / surv
    m1 = max(0, 80 - c1)
    if m1 == 0 or k < c1 + 20:
        return 0.0
    p2 = (m1 / NPAIR) / (1.0 / 80.0 + m1 / NPAIR)
    return (p2 / m1) / ((1.0 - p2) + p2 * (100 - k) / m1)


def haz_sp(k, n, c1):
    """P(setpoint change at sample k | history); exactly two changes per batch."""
    if k < C_LO or k >= C_HI or n >= 2:
        return 0.0
    if n == 0:
        return 2.0 / (81 - k) if k <= 79 else 1.0
    return 1.0 / (100 - k) if k >= c1 + 20 else 0.0


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
        self.X = None
        self.sp_prev, self.sp_n, self.sp_c1 = None, 0, 0
        self.H = int(p["H"])
        blocks = list(p["blocks"])
        assert sum(blocks) == self.H
        self.M = len(blocks)
        self.bmap = [i for i, b in enumerate(blocks) for _ in range(b)]
        self.nb1 = sum(1 for b in blocks if b == 1)
        self.v = np.full(self.M, 299.5)    # decision vector (block values), warm start

    # ---------------- estimator ----------------
    # Hypotheses differ in when the feed shifts happened. Each carries an EKF mean/covariance
    # for [Ca, T, Ti, Caf] plus (number of shifts, time of first shift) for Ti and for Caf.
    def _init_est(self, z):
        p = self.p
        self.X = np.array([[z[0], z[1], 350.0, 1.0]])
        self.PP = np.diag([p["r_ca"] ** 2, p["r_T"] ** 2, p["p0_ti"] ** 2, p["p0_caf"] ** 2])[None]
        self.lw = np.zeros(1)
        self.meta = np.zeros((1, 4), dtype=int)      # nTi, c1Ti, nCaf, c1Caf

    def _measure(self, z):
        X, P = self.X, self.PP
        r1, r2 = self.p["r_ca"] ** 2, self.p["r_T"] ** 2
        s11 = P[:, 0, 0] + r1; s12 = P[:, 0, 1]; s22 = P[:, 1, 1] + r2
        det = s11 * s22 - s12 * s12
        i11, i12, i22 = s22 / det, -s12 / det, s11 / det
        e1 = z[0] - X[:, 0]; e2 = z[1] - X[:, 1]
        self.lw = self.lw - 0.5 * (e1 * e1 * i11 + 2 * e1 * e2 * i12 + e2 * e2 * i22) - 0.5 * np.log(det)
        K1 = P[:, :, 0] * i11[:, None] + P[:, :, 1] * i12[:, None]     # (n,4) gain column for Ca
        K2 = P[:, :, 0] * i12[:, None] + P[:, :, 1] * i22[:, None]
        self.X = X + K1 * e1[:, None] + K2 * e2[:, None]
        P = P - K1[:, :, None] * P[:, None, 0, :] - K2[:, :, None] * P[:, None, 1, :]
        self.PP = 0.5 * (P + P.transpose(0, 2, 1))
        # prune to the n_hyp most likely hypotheses
        lw = self.lw - self.lw.max()
        n = int(self.p["n_hyp"])
        if len(lw) > n:
            keep = np.argsort(lw)[-n:]
            self.X, self.PP, self.meta, lw = self.X[keep], self.PP[keep], self.meta[keep], lw[keep]
        self.lw = lw

    def _estimate(self):
        w = np.exp(self.lw)
        w /= w.sum()
        x = w @ self.X
        x[2] = min(max(x[2], 348.5), 351.5)
        x[3] = min(max(x[3], 0.98), 1.02)
        return x

    def _predict(self, u, xbar):
        p = self.p
        ca, T, Ti, Caf = (float(v) for v in xbar)
        c1, T1 = _step(ca, T, u, Ti, Caf, 4)
        F = np.eye(4)
        for i, e in enumerate((1e-6, 1e-3, 1e-3, 1e-6)):
            a = [ca, T, Ti, Caf]
            a[i] += e
            c2, T2 = _step(a[0], a[1], u, a[2], a[3], 4)
            F[0, i] = (c2 - c1) / e
            F[1, i] = (T2 - T1) / e
        X = np.array([c1, T1, Ti, Caf]) + (self.X - xbar) @ F.T
        P = F @ self.PP @ F.T
        P[:, 0, 0] += p["q_ca"] ** 2
        P[:, 1, 1] += p["q_T"] ** 2
        # branch on a feed shift at sample index k+1 (the value acting over the next interval)
        kj = self.k + 1
        meta = self.meta
        hs = p["haz_scale"]
        hT = np.array([min(1.0, hs * haz_dist(kj, m[0], m[1])) for m in meta])
        hC = np.array([min(1.0, hs * haz_dist(kj, m[2], m[3])) for m in meta])
        if hT.max() <= 0.0 and hC.max() <= 0.0:
            self.X, self.PP = X, P
            return
        Xs, Ps, ms, ls = [], [], [], []
        for jT in (0, 1):
            for jC in (0, 1):
                pr = (hT if jT else 1.0 - hT) * (hC if jC else 1.0 - hC)
                ok = pr > 1e-9
                if not ok.any():
                    continue
                Xb, Pb, mb = X[ok].copy(), P[ok].copy(), meta[ok].copy()
                if jT:
                    Xb[:, 2] = 350.0
                    Pb[:, 2, :] = 0.0
                    Pb[:, :, 2] = 0.0
                    Pb[:, 2, 2] = p["p0_ti"] ** 2
                    mb[:, 1] = np.where(mb[:, 0] == 0, kj, mb[:, 1])
                    mb[:, 0] += 1
                if jC:
                    Xb[:, 3] = 1.0
                    Pb[:, 3, :] = 0.0
                    Pb[:, :, 3] = 0.0
                    Pb[:, 3, 3] = p["p0_caf"] ** 2
                    mb[:, 3] = np.where(mb[:, 2] == 0, kj, mb[:, 3])
                    mb[:, 2] += 1
                Xs.append(Xb)
                Ps.append(Pb)
                ms.append(mb)
                ls.append(self.lw[ok] + np.log(pr[ok]))
        self.X, self.PP = np.concatenate(Xs), np.concatenate(Ps)
        self.meta, self.lw = np.concatenate(ms), np.concatenate(ls)

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

    def _mpc(self, ca0, T0, Ti, Caf, target):
        H = min(self.H, NSTEP - self.k)
        M = self.bmap[H - 1] + 1
        v = self.v[:M].copy()
        mu = self.p["mu"]
        best_v, best_c = v.copy(), np.inf
        iters = self.p["gn_iters"]
        for it in range(iters + 1):
            last = it == iters
            out, J = self._lin_rollout(ca0, T0, [float(a) for a in v], Ti, Caf, H, M, jac=not last)
            r = out - target[:H]
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

    def _targets(self, sp):
        """Per-stage targets: the setpoint, pulled towards the mean of a possible new setpoint
        (0.88) in proportion to the chance that a setpoint change arrives at that sample."""
        c = self.p["hedge_c"]
        tg = np.full(self.H, sp)
        if c > 0.0:
            for j in range(self.H):
                h = haz_sp(self.k + j + 1, self.sp_n, self.sp_c1)
                if h > 0.0:
                    tg[j] = sp + (h * c / (1.0 + h * c)) * (0.88 - sp)
        return tg

    def act(self, obs):
        z = (obs["Ca"], obs["T"])
        sp = obs["Ca_sp"]
        if self.sp_prev is not None and sp != self.sp_prev:
            if self.sp_n == 0:
                self.sp_c1 = self.k
            self.sp_n += 1
        self.sp_prev = sp
        if self.X is None:
            self._init_est(z)
        else:
            self._measure(z)
        x = self._estimate()
        if self.p["oracle"] and "Ti_true" in obs:
            x = np.array([obs["Ca"], obs["T"], obs["Ti_true"], obs["Caf_true"]])
        ca, T, Ti, Caf = (float(a) for a in x)
        u = self._mpc(ca, T, Ti, Caf, self._targets(sp))
        u = min(max(u, ULO), UHI)
        self._predict(u, x)
        self.v[:self.nb1 - 1] = self.v[1:self.nb1]   # warm start: shift the single-step blocks
        self.k += 1
        return u
