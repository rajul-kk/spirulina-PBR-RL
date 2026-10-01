"""Model-based controller for the jacketed CSTR: extended Kalman filter + nonlinear MPC.

The filter estimates the true concentration and temperature and the two unmeasured feed
disturbances (feed temperature Ti, feed concentration Caf) from the noisy Ca and T measurements,
using the plant's design equations. The MPC then picks the jacket temperature sequence that
minimises the predicted squared Ca error over the next few minutes, within the actuator limits.
"""
import math

import numpy as np

K0, EA = 7.2e10, 8750.0
DH = 5e4 / (1000 * 0.239)            # (-dHr)/(rho C)
UA = 5e4 / (1000 * 0.239 * 100)      # UA/(rho C V)
DT = 26.0 / 120.0                    # sample time, min
U_LO, U_HI = 295.0, 302.0

DEFAULTS = {
    "q_ti": 1e-3,        # slow-drift variance per sample for Ti, K^2
    "q_caf": 1e-7,       # same for Caf, (mol/L)^2
    "q_ca": 1e-10, "q_T": 1e-6,
    "r_ca": 0.002 ** 2, "r_T": 0.2 ** 2,
    "p0_ti": 0.75, "p0_caf": 1.33e-4,
    "blocks": (1, 1, 1, 1, 2, 2),   # move blocking of the free part of the plan
    "horizon": 15,
    "iters": 2,
    "rho": 0.0,          # penalty on plan moves (per K^2, in cost units)
    "t_lim": 333.0, "w_t": 5.0,           # soft limit on predicted reactor temperature
    "d_margin": 0.005,    # disturbance estimates are kept within the feed spec range +- margin
    "clip_d": 1,
    "p_jump": 0.01,      # per-sample probability of a step change in each feed disturbance
    "p_jump_out": 0.0005, "use_window": 1,   # changes are rare in the first/last 20 samples
    "jump_ti": 1.5, "jump_caf": 2.67e-4,   # variance of a step change
}


def _rhs(ca, T, tc, ti, caf):
    r = K0 * np.exp(-EA / T) * ca
    return caf - ca - r, ti - T + DH * r + UA * (tc - T)


def _rk4(ca, T, tc, ti, caf, h):
    a1, b1 = _rhs(ca, T, tc, ti, caf)
    a2, b2 = _rhs(ca + 0.5 * h * a1, T + 0.5 * h * b1, tc, ti, caf)
    a3, b3 = _rhs(ca + 0.5 * h * a2, T + 0.5 * h * b2, tc, ti, caf)
    a4, b4 = _rhs(ca + h * a3, T + h * b3, tc, ti, caf)
    return (ca + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4),
            T + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4))


def steady_tc(sp, ti, caf):
    """Jacket temperature that holds Ca = sp at steady state (unclipped)."""
    k = max(caf / sp - 1.0, 1e-6)
    T = EA / math.log(K0 / k)
    return T - (ti - T + DH * k * sp) / UA


def _box_qp(Hm, g, lo, hi):
    """min 0.5 d'Hd + g'd  s.t. lo <= d <= hi (lo <= 0 <= hi); primal active-set method."""
    n = len(g)
    d = np.zeros(n)
    gr = g.copy()
    fixed = ((d <= lo) & (gr > 0)) | ((d >= hi) & (gr < 0))
    for _ in range(10 * n):
        fi = np.where(~fixed)[0]
        s = np.zeros(n)
        if len(fi):
            s[fi] = np.linalg.solve(Hm[np.ix_(fi, fi)], -gr[fi])
        if len(fi) == 0 or np.max(np.abs(s)) < 1e-8:
            # minimiser on the current face: release the bound with the most wrong-signed multiplier
            viol = np.where(fixed, np.where(d <= lo, -gr, gr), 0.0)
            viol[(d <= lo) & (d >= hi)] = 0.0
            j = int(np.argmax(viol))
            if viol[j] <= 1e-9:
                break
            fixed[j] = False
            continue
        alpha, jb = 1.0, -1
        for i in fi:
            if s[i] > 0:
                a = (hi[i] - d[i]) / s[i]
            elif s[i] < 0:
                a = (lo[i] - d[i]) / s[i]
            else:
                continue
            if a < alpha:
                alpha, jb = a, i
        d = np.minimum(np.maximum(d + alpha * s, lo), hi)
        if jb >= 0:
            d[jb] = hi[jb] if s[jb] > 0 else lo[jb]
            fixed[jb] = True
        gr = g + Hm @ d
    return d


class Controller:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.z = None                      # [Ca, T, Ti, Caf]
        self.P = None
        self.u_prev = 298.5
        self.plan = None
        self.sp_prev = None
        self.k = 0
        self.n_fallback = 0
        self.Q = np.diag([p["q_ca"], p["q_T"], p["q_ti"], p["q_caf"]])
        self.R = np.diag([p["r_ca"], p["r_T"]])
        self.blocks = tuple(int(b) for b in p["blocks"])
        self.H = int(p["horizon"])
        idx = []
        for i, b in enumerate(self.blocks):
            idx += [i] * b
        self.nfree = len(idx)
        self.idx = np.array(idx[: self.H])
        self.M = len(self.blocks)

    # ---------------- estimator ----------------
    def _ekf(self, y_ca, y_T):
        """EKF on [Ca, T, Ti, Caf]. The feed disturbances are constant between rare step changes, so
        each sample is filtered under three hypotheses (no change / Ti stepped / Caf stepped), which
        are weighted by how well they explain the new measurement and merged (GPB1)."""
        p = self.p
        if self.z is None:
            self.z = np.array([y_ca, y_T, 350.0, 1.0])
            self.P = np.diag([p["r_ca"], p["r_T"], p["p0_ti"], p["p0_caf"]])
            return
        ca, T, ti, caf = self.z
        k = K0 * math.exp(-EA / T)
        g = k * ca * EA / (T * T)
        A = np.zeros((4, 4))
        A[0, 0], A[0, 1], A[0, 3] = -(1.0 + k), -g, 1.0
        A[1, 0], A[1, 1], A[1, 2] = DH * k, -1.0 - UA + DH * g, 1.0
        Ad = A * DT
        I4 = np.eye(4)
        F = I4 + Ad @ (I4 + Ad @ (I4 / 2 + Ad @ (I4 / 6 + Ad / 24)))
        c, t = float(ca), float(T)
        for _ in range(2):
            c, t = _rk4(c, t, self.u_prev, ti, caf, DT / 2)
        zp = np.array([c, t, ti, caf])
        y = np.array([y_ca, y_T])
        innov = y - zp[:2]
        # a disturbance change at this sample acts during the sample: add it before propagating
        pj = p["p_jump"]
        if p["use_window"] and not (20 <= self.k - 1 < 100):
            pj = p["p_jump_out"]
        hyps = [(1.0 - 2 * pj, 0.0, 0.0)]
        if pj > 0:
            hyps += [(pj, p["jump_ti"], 0.0), (pj, 0.0, p["jump_caf"])]
        zs, Ps, ws = [], [], []
        for w, jt, jc in hyps:
            P0 = self.P.copy()
            P0[2, 2] += p["q_ti"] + jt
            P0[3, 3] += p["q_caf"] + jc
            P = F @ P0 @ F.T
            P[0, 0] += p["q_ca"]
            P[1, 1] += p["q_T"]
            S = P[:2, :2] + self.R
            Si = np.linalg.inv(S)
            K = P[:, :2] @ Si
            IKH = I4.copy()
            IKH[:, :2] -= K
            zs.append(zp + K @ innov)
            Ps.append(IKH @ P @ IKH.T + K @ self.R @ K.T)
            ws.append(math.log(max(w, 1e-300)) - 0.5 * float(innov @ Si @ innov)
                      - 0.5 * math.log(np.linalg.det(S)))
        ws = np.exp(np.array(ws) - max(ws))
        ws = ws / ws.sum()
        z = sum(w * zz for w, zz in zip(ws, zs))
        P = np.zeros((4, 4))
        for w, zz, PP in zip(ws, zs, Ps):
            dz = (zz - z)[:, None]
            P += w * (PP + dz @ dz.T)
        if p["clip_d"]:
            m = p["d_margin"]
            z[2] = min(max(z[2], 348.5 - 75 * m), 351.5 + 75 * m)
            z[3] = min(max(z[3], 0.98 - m), 1.02 + m)
        self.z, self.P = z, P

    # ---------------- predictor / optimiser ----------------
    def _predict(self, ca0, T0, U, ti, caf):
        """U: (P, H) jacket sequences -> predicted Ca, T of shape (P, H)."""
        npert = U.shape[0]
        ca = np.full(npert, ca0)
        T = np.full(npert, T0)
        CA = np.empty_like(U)
        TT = np.empty_like(U)
        for j in range(U.shape[1]):
            ca, T = _rk4(ca, T, U[:, j], ti, caf, DT)
            CA[:, j] = ca
            TT[:, j] = T
        return CA, TT

    def _expand(self, V, u_tail):
        """V: (P, M) block values -> (P, H) sequences with the tail held at u_tail."""
        U = np.full((V.shape[0], self.H), u_tail)
        n = len(self.idx)
        U[:, :n] = V[:, self.idx]
        return U

    def _resid(self, V, u_tail, ca0, T0, ti, caf, sp):
        p = self.p
        CA, TT = self._predict(ca0, T0, self._expand(V, u_tail), ti, caf)
        parts = [(CA - sp) / 0.01, p["w_t"] * np.maximum(TT - p["t_lim"], 0.0)]
        if p["rho"] > 0:
            s = math.sqrt(p["rho"])
            dV = np.diff(np.concatenate([np.full((V.shape[0], 1), self.u_prev), V], axis=1), axis=1)
            parts.append(s * dV)
        return np.concatenate(parts, axis=1)

    def _mpc(self, ca0, T0, ti, caf, sp):
        M = self.M
        u_tail = min(max(steady_tc(sp, ti, caf), U_LO), U_HI)
        if self.plan is None or sp != self.sp_prev:
            v = np.full(M, u_tail)
        else:
            v = self.plan.copy()
        eps = 0.02
        for _ in range(int(self.p["iters"])):
            V = np.tile(v, (M + 1, 1))
            V[1:] += eps * np.eye(M)
            Rr = self._resid(V, u_tail, ca0, T0, ti, caf, sp)
            r = Rr[0]
            J = (Rr[1:] - r).T / eps
            c0 = float(r @ r)
            Hm = J.T @ J + 1e-6 * np.eye(M)
            g = J.T @ r
            d = _box_qp(Hm, g, U_LO - v, U_HI - v)
            # backtracking on the true (nonlinear) cost
            best, bc = v, c0
            cand = np.array([np.clip(v + a * d, U_LO, U_HI) for a in (1.0, 0.5, 0.25)])
            Rc = self._resid(cand, u_tail, ca0, T0, ti, caf, sp)
            cc = np.sum(Rc * Rc, axis=1)
            i = int(np.argmin(cc))
            if cc[i] < bc:
                best, bc = cand[i], cc[i]
            if c0 - bc < 1e-9:
                v = best
                break
            v = best
        self.plan = v
        self.sp_prev = sp
        return float(v[0])

    # ---------------- main entry ----------------
    def act(self, obs):
        try:
            u = self._act(obs)
            if u == u:
                return u
        except Exception:
            pass
        self.n_fallback += 1
        # fallback (never expected): steady-state jacket value for the last estimate, with a
        # proportional correction on the measured Ca error (higher Ca -> warmer jacket)
        sp = float(obs["Ca_sp"])
        ti, caf = (350.0, 1.0) if self.z is None else (float(self.z[2]), float(self.z[3]))
        u = steady_tc(sp, ti, caf) + 100.0 * (float(obs["Ca"]) - sp)
        u = min(max(u, U_LO), U_HI)
        self.u_prev = u
        self.plan = None
        return u

    def _act(self, obs):
        sp = float(obs["Ca_sp"])
        if "_truth" in obs:                       # offline studies only (my own simulator)
            ca, T, ti, caf = obs["_truth"]
        else:
            self._ekf(float(obs["Ca"]), float(obs["T"]))
            ca, T, ti, caf = (float(x) for x in self.z)
        if self.plan is not None and sp == self.sp_prev:
            # shift the plan by one sample (blocks of length 1 shift exactly)
            v = self.plan
            nv = v.copy()
            nv[:-1] = np.where(np.array(self.blocks[:-1]) == 1, v[1:], v[:-1])
            self.plan = nv
        u = self._mpc(ca, T, ti, caf, sp)
        u = min(max(u, U_LO), U_HI)
        self.u_prev = u
        self.k += 1
        return u
