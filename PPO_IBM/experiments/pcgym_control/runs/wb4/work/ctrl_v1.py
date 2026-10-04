"""Jacketed CSTR controller: multi-hypothesis extended Kalman filter + nonlinear MPC.

Estimator: the plant model is known, so the only unknowns are the measurement noise and the two
unmeasured feed disturbances (feed temperature Ti, feed concentration Caf), which are piecewise
constant with rare steps. A small bank of extended Kalman filters is kept, one per hypothesis on
when the feed last stepped ("no step", "Ti stepped at sample j", "Caf stepped at sample j", ...).
Each is weighted by how well it explains the measurements, and the weighted mean is used.

Control: nonlinear model-predictive control. At each sample the jacket-temperature sequence over
the horizon is optimised (Gauss-Newton with box constraints) to minimise the predicted squared
concentration error, with a soft ceiling on the reactor temperature.
"""
import math

import numpy as np

# ---- plant model (PC-Gym cstr_ode; time in minutes) ----
K0 = 7.2e10
EA = 8750.0
DH = 5e4 / (1000.0 * 0.239)            # K per (mol/L): heat of reaction / (rho*C)
UA = 5e4 / (1000.0 * 0.239 * 100.0)    # 1/min: jacket heat-transfer rate
DT = 26.0 / 120.0                      # sample time, min
U_LO, U_HI = 295.0, 302.0
TI_LO, TI_HI = 348.5, 351.5
CAF_LO, CAF_HI = 0.98, 1.02
N_STEPS = 120
SEG_MIN = 20                           # minimum number of samples between feed steps

DEFAULTS = {
    "H": 14,            # prediction horizon, samples
    "nsub": 4,          # RK4 substeps per sample
    "rho": 1e-3,        # move penalty (cost units per K^2)
    "n_iter": 3,        # Gauss-Newton iterations per sample
    "T_max": 331.0,     # soft ceiling on predicted reactor temperature, K
    "w_T": 50.0,        # weight of the ceiling violation
    "w_term": 0.0,      # terminal weight on (T - T_ss)
    "M": 8,             # hypotheses kept in the filter bank
    "r_ca": 0.002,      # analyser noise std
    "r_T": 0.2,         # temperature sensor noise std
    "q_ca": 1e-5,       # process noise std (model integration error), Ca
    "q_T": 1e-3,        # process noise std, T
    "lam": 1e-6,        # Levenberg damping
}


def _step(ca, T, tc, ti, caf, nsub):
    """One sample of the plant with scalar RK4."""
    h = DT / nsub
    exp = math.exp
    for _ in range(nsub):
        r = K0 * exp(-EA / T) * ca
        a1 = caf - ca - r
        b1 = ti - T + DH * r + UA * (tc - T)
        c2 = ca + 0.5 * h * a1
        T2 = T + 0.5 * h * b1
        r = K0 * exp(-EA / T2) * c2
        a2 = caf - c2 - r
        b2 = ti - T2 + DH * r + UA * (tc - T2)
        c3 = ca + 0.5 * h * a2
        T3 = T + 0.5 * h * b2
        r = K0 * exp(-EA / T3) * c3
        a3 = caf - c3 - r
        b3 = ti - T3 + DH * r + UA * (tc - T3)
        c4 = ca + h * a3
        T4 = T + h * b3
        r = K0 * exp(-EA / T4) * c4
        a4 = caf - c4 - r
        b4 = ti - T4 + DH * r + UA * (tc - T4)
        ca = ca + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
        T = T + h * (b1 + 2 * b2 + 2 * b3 + b4) / 6.0
    return ca, T


def _step_v(ca, T, tc, ti, caf, nsub):
    """Vectorised version of _step (numpy arrays)."""
    h = DT / nsub
    for _ in range(nsub):
        r = K0 * np.exp(-EA / T) * ca
        a1 = caf - ca - r
        b1 = ti - T + DH * r + UA * (tc - T)
        c2 = ca + 0.5 * h * a1
        T2 = T + 0.5 * h * b1
        r = K0 * np.exp(-EA / T2) * c2
        a2 = caf - c2 - r
        b2 = ti - T2 + DH * r + UA * (tc - T2)
        c3 = ca + 0.5 * h * a2
        T3 = T + 0.5 * h * b2
        r = K0 * np.exp(-EA / T3) * c3
        a3 = caf - c3 - r
        b3 = ti - T3 + DH * r + UA * (tc - T3)
        c4 = ca + h * a3
        T4 = T + h * b3
        r = K0 * np.exp(-EA / T4) * c4
        a4 = caf - c4 - r
        b4 = ti - T4 + DH * r + UA * (tc - T4)
        ca = ca + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
        T = T + h * (b1 + 2 * b2 + 2 * b3 + b4) / 6.0
    return ca, T


_E_CA, _E_T, _E_U, _E_TI, _E_CAF = 1e-5, 1e-3, 1e-3, 1e-3, 1e-5


def _jac_xu(ca, T, tc, ti, caf, nsub):
    """Discrete-time Jacobians along a trajectory. Inputs are arrays of length n.
    Returns A (n,2,2) = d x+/d x and B (n,2) = d x+/d Tc (central differences)."""
    n = len(ca)
    c = np.repeat(ca[:, None], 6, axis=1)
    t = np.repeat(T[:, None], 6, axis=1)
    u = np.repeat(tc[:, None], 6, axis=1)
    c[:, 0] += _E_CA
    c[:, 1] -= _E_CA
    t[:, 2] += _E_T
    t[:, 3] -= _E_T
    u[:, 4] += _E_U
    u[:, 5] -= _E_U
    cn, tn = _step_v(c, t, u, ti, caf, nsub)
    A = np.empty((n, 2, 2))
    A[:, 0, 0] = (cn[:, 0] - cn[:, 1]) / (2 * _E_CA)
    A[:, 1, 0] = (tn[:, 0] - tn[:, 1]) / (2 * _E_CA)
    A[:, 0, 1] = (cn[:, 2] - cn[:, 3]) / (2 * _E_T)
    A[:, 1, 1] = (tn[:, 2] - tn[:, 3]) / (2 * _E_T)
    B = np.empty((n, 2))
    B[:, 0] = (cn[:, 4] - cn[:, 5]) / (2 * _E_U)
    B[:, 1] = (tn[:, 4] - tn[:, 5]) / (2 * _E_U)
    return A, B


def _jac_full(ca, T, tc, ti, caf, nsub):
    """4x4 Jacobian of the augmented state [Ca, T, Ti, Caf] over one sample."""
    eps = np.array([_E_CA, _E_T, _E_TI, _E_CAF])
    z = np.array([ca, T, ti, caf])
    Z = np.repeat(z[:, None], 8, axis=1)
    for i in range(4):
        Z[i, 2 * i] += eps[i]
        Z[i, 2 * i + 1] -= eps[i]
    cn, tn = _step_v(Z[0], Z[1], tc, Z[2], Z[3], nsub)
    F = np.eye(4)
    for i in range(4):
        F[0, i] = (cn[2 * i] - cn[2 * i + 1]) / (2 * eps[i])
        F[1, i] = (tn[2 * i] - tn[2 * i + 1]) / (2 * eps[i])
    return F


# ---- prior on when the feed steps (from the scenario definition) ----
# Each feed signal has 1 or 2 steps (equally likely) at samples 20..99, segments >= 20 samples.
_N_PAIRS = 1830.0   # number of valid (first, second) step-time pairs


def _p_first(j):
    if j < SEG_MIN or j > 99:
        return 0.0
    p = 0.5 / 80.0
    if j <= 79:
        p += 0.5 * (80 - j) / _N_PAIRS
    return p


_SURV = [0.0] * (N_STEPS + 2)
for _j in range(N_STEPS, -1, -1):
    _SURV[_j] = _SURV[_j + 1] + _p_first(_j)


def _hazard(j, n, last):
    """Probability that a feed signal steps at sample j given its history (n steps so far,
    the latest at sample `last`) and no step since."""
    if j < SEG_MIN or j > 99 or n >= 2:
        return 0.0
    if n == 0:
        s = _SURV[j]
        return min(1.0, _p_first(j) / s) if s > 0 else 0.0
    c = last
    if c > 79 or j < c + SEG_MIN:
        return 0.0
    w2 = (0.5 * (80 - c) / _N_PAIRS) / _p_first(c)
    den = 1.0 - w2 * (j - c - SEG_MIN) / (80.0 - c)
    return min(1.0, (w2 / (80.0 - c)) / den) if den > 0 else 0.0


def _box_qp(Hm, g, lo, hi, iters=25):
    """min 0.5 x'Hx + g'x  s.t. lo <= x <= hi  (projected Newton)."""
    n = len(g)
    x = np.clip(np.zeros(n), lo, hi)
    f = 0.5 * x @ Hm @ x + g @ x
    for _ in range(iters):
        q = Hm @ x + g
        active = ((x <= lo + 1e-12) & (q > 0)) | ((x >= hi - 1e-12) & (q < 0))
        free = ~active
        if not free.any():
            break
        dx = np.zeros(n)
        idx = np.where(free)[0]
        try:
            dx[idx] = np.linalg.solve(Hm[np.ix_(idx, idx)], -q[idx])
        except np.linalg.LinAlgError:
            dx[idx] = -q[idx] / np.diag(Hm)[idx]
        a = 1.0
        improved = False
        for _ls in range(12):
            xn = np.clip(x + a * dx, lo, hi)
            fn = 0.5 * xn @ Hm @ xn + g @ xn
            if fn < f - 1e-14:
                improved = True
                break
            a *= 0.5
        if not improved:
            break
        done = np.max(np.abs(xn - x)) < 1e-7
        x, f = xn, fn
        if done:
            break
    return x


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.k = 0
        self.u_prev = None
        self.u_plan = None
        self.hyps = None        # list of [w, m(4), P(4x4), nTi, lastTi, nCaf, lastCaf]
        self.R = np.diag([p["r_ca"] ** 2, p["r_T"] ** 2])
        self.Q = np.diag([p["q_ca"] ** 2, p["q_T"] ** 2, 0.0, 0.0])
        self.est = None

    # ------------------------------------------------------------------ estimator
    def _estimate(self, obs):
        p = self.p
        y = np.array([obs["Ca"], obs["T"]])
        if self.hyps is None:
            m = np.array([y[0], y[1], 0.5 * (TI_LO + TI_HI), 0.5 * (CAF_LO + CAF_HI)])
            P = np.diag([p["r_ca"] ** 2, p["r_T"] ** 2,
                         (TI_HI - TI_LO) ** 2 / 12.0, (CAF_HI - CAF_LO) ** 2 / 12.0])
            self.hyps = [[1.0, m, P, 0, -1, 0, -1]]
            return m.copy()
        j = self.k - 1                      # sample index of the interval just completed
        u = self.u_prev
        mm = self._merged()
        F = _jac_full(mm[0], mm[1], u, mm[2], mm[3], p["nsub"])
        var_ti = (TI_HI - TI_LO) ** 2 / 12.0
        var_caf = (CAF_HI - CAF_LO) ** 2 / 12.0
        new = []
        for w, m, P, nti, lti, ncaf, lcaf in self.hyps:
            h_ti = _hazard(j, nti, lti)
            h_caf = _hazard(j, ncaf, lcaf)
            for s_ti in (0, 1):
                pw_ti = h_ti if s_ti else 1.0 - h_ti
                if pw_ti <= 0.0:
                    continue
                for s_caf in (0, 1):
                    pw = pw_ti * (h_caf if s_caf else 1.0 - h_caf)
                    if pw <= 0.0:
                        continue
                    m2, P2 = m, P
                    if s_ti or s_caf:
                        m2 = m.copy()
                        P2 = P.copy()
                        if s_ti:
                            m2[2] = 0.5 * (TI_LO + TI_HI)
                            P2[2, :] = 0.0
                            P2[:, 2] = 0.0
                            P2[2, 2] = var_ti
                        if s_caf:
                            m2[3] = 0.5 * (CAF_LO + CAF_HI)
                            P2[3, :] = 0.0
                            P2[:, 3] = 0.0
                            P2[3, 3] = var_caf
                    ca, T = _step(m2[0], m2[1], u, m2[2], m2[3], p["nsub"])
                    mp = np.array([ca, T, m2[2], m2[3]])
                    Pp = F @ P2 @ F.T + self.Q
                    S = Pp[:2, :2] + self.R
                    Sinv = np.linalg.inv(S)
                    innov = y - mp[:2]
                    K = Pp[:, :2] @ Sinv
                    mu = mp + K @ innov
                    Pu = Pp - K @ Pp[:2, :]
                    Pu = 0.5 * (Pu + Pu.T)
                    lik = math.exp(-0.5 * float(innov @ Sinv @ innov)) / math.sqrt(
                        max(np.linalg.det(S), 1e-300))
                    new.append([w * pw * lik, mu, Pu,
                                nti + s_ti, j if s_ti else lti,
                                ncaf + s_caf, j if s_caf else lcaf])
        tot = sum(h[0] for h in new)
        if not (tot > 0.0) or not math.isfinite(tot):
            # every hypothesis is numerically impossible: fall back to the predictions, equal weights
            for h in new:
                h[0] = 1.0
            tot = float(len(new))
        new.sort(key=lambda h: -h[0])
        new = new[: p["M"]]
        tot = sum(h[0] for h in new)
        for h in new:
            h[0] /= tot
        self.hyps = [h for h in new if h[0] > 1e-9]
        tot = sum(h[0] for h in self.hyps)
        for h in self.hyps:
            h[0] /= tot
        return self._merged()

    def _merged(self):
        m = np.zeros(4)
        for h in self.hyps:
            m += h[0] * h[1]
        return m

    # ------------------------------------------------------------------ controller
    def _rollout(self, ca, T, useq, ti, caf):
        n = len(useq)
        cas = np.empty(n + 1)
        Ts = np.empty(n + 1)
        cas[0], Ts[0] = ca, T
        nsub = self.p["nsub"]
        for i in range(n):
            ca, T = _step(ca, T, useq[i], ti, caf, nsub)
            cas[i + 1], Ts[i + 1] = ca, T
        return cas, Ts

    def _residuals(self, cas, Ts, useq, sp, u_last, T_ss):
        p = self.p
        r_ca = (cas[1:] - sp) / 0.01
        du = np.diff(np.concatenate([[u_last], useq]))
        r_du = math.sqrt(p["rho"]) * du
        over = np.maximum(Ts[1:] - p["T_max"], 0.0)
        r_T = math.sqrt(p["w_T"]) * over
        r_term = math.sqrt(p["w_term"]) * (Ts[-1] - T_ss)
        return r_ca, r_du, r_T, r_term

    def _mpc(self, ca, T, ti, caf, sp):
        p = self.p
        H = min(p["H"], max(2, N_STEPS - self.k + 1))
        u_last = self.u_prev if self.u_prev is not None else 0.5 * (U_LO + U_HI)
        if self.u_plan is None:
            useq = np.full(H, self._u_ss(sp, ti, caf))
        else:
            useq = np.concatenate([self.u_plan[1:], self.u_plan[-1:]])
            if len(useq) < H:
                useq = np.concatenate([useq, np.full(H - len(useq), useq[-1])])
            useq = useq[:H]
        useq = np.clip(useq, U_LO, U_HI)
        kk = max((caf - sp) / sp, 1e-6)
        T_ss = EA / math.log(K0 / kk)
        cas, Ts = self._rollout(ca, T, useq, ti, caf)
        res = self._residuals(cas, Ts, useq, sp, u_last, T_ss)
        cost = sum(float(np.sum(r * r)) for r in res)
        D = np.eye(H) - np.eye(H, k=-1)
        sr, swT, swt = math.sqrt(p["rho"]), math.sqrt(p["w_T"]), math.sqrt(p["w_term"])
        for _ in range(p["n_iter"]):
            A, B = _jac_xu(cas[:-1], Ts[:-1], useq, ti, caf, p["nsub"])
            # sensitivities of x_{i+1} to u_j
            Gc = np.zeros((H, H))
            Gt = np.zeros((H, H))
            for jx in range(H):
                s = B[jx]
                Gc[jx, jx], Gt[jx, jx] = s[0], s[1]
                for i in range(jx + 1, H):
                    s = A[i] @ s
                    Gc[i, jx], Gt[i, jx] = s[0], s[1]
            r_ca, r_du, r_T, r_term = res
            J = [Gc / 0.01, sr * D]
            r = [r_ca, r_du]
            act = r_T > 0
            if act.any():
                J.append(swT * Gt[act])
                r.append(r_T[act])
            if swt > 0:
                J.append(swt * Gt[-1:])
                r.append(np.array([r_term]))
            J = np.vstack(J)
            r = np.concatenate(r)
            Hm = J.T @ J + p["lam"] * np.eye(H)
            g = J.T @ r
            dx = _box_qp(Hm, g, U_LO - useq, U_HI - useq)
            if np.max(np.abs(dx)) < 1e-5:
                break
            a = 1.0
            ok = False
            for _ls in range(5):
                un = np.clip(useq + a * dx, U_LO, U_HI)
                cn, tn = self._rollout(ca, T, un, ti, caf)
                rn = self._residuals(cn, tn, un, sp, u_last, T_ss)
                costn = sum(float(np.sum(q * q)) for q in rn)
                if costn < cost:
                    ok = True
                    break
                a *= 0.5
            if not ok:
                break
            conv = cost - costn < 1e-7
            useq, cas, Ts, res, cost = un, cn, tn, rn, costn
            if conv:
                break
        self.u_plan = useq
        return float(useq[0])

    @staticmethod
    def _u_ss(sp, ti, caf):
        """Jacket temperature that holds Ca = sp at steady state."""
        kk = max((caf - sp) / sp, 1e-6)
        T = EA / math.log(K0 / kk)
        r = kk * sp
        return T - (ti - T + DH * r) / UA

    def act(self, obs):
        z = self._estimate(obs)
        self.est = z
        ca = float(z[0])
        T = float(z[1])
        ti = float(min(max(z[2], TI_LO), TI_HI))
        caf = float(min(max(z[3], CAF_LO), CAF_HI))
        u = self._mpc(ca, T, ti, caf, float(obs["Ca_sp"]))
        u = float(min(max(u, U_LO), U_HI))
        self.u_prev = u
        self.k += 1
        return u
