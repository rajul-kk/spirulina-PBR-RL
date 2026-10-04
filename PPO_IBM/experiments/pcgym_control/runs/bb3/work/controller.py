"""Ca controller for the jacketed CSTR (run bb3).

Structure: extended Kalman filter on [Ca, T, Caf, Tf] (feed concentration and feed temperature are
unmeasured, estimated as random-walk states with an innovation test that reopens them after a
feed shift) + nonlinear model-predictive control with move blocking, solved by grid search and
local refinement.  Model identified from pilot batches (prediction-error fit through the EKF):
    dCa/dt = a (Caf - Ca) - k(T) Ca
    dT/dt  = a (Tf - T) + b k(T) Ca + c (Tc - T),   k(T) = kr exp(-ER (1/T - 1/323))
Only numpy, math (and collections) are used; no file/OS/network access.
"""
import math
import numpy as np

DT = 13.0 / 60.0  # min per sample

DEFAULTS = dict(
    # identified model (33 pilot batches); Caf, Tf are the nominal feed values in this parametrisation
    a=0.73716, kr=0.11696, ER=8775.66, b=190.579, c=2.12598, Caf=1.032, Tf=365.2, Tref=323.0,
    # estimator: analyser / sensor noise, process noise, feed-shift random walk, jump detection
    r_ca=0.002, r_t=0.2, q_ca=1e-4, q_t=1e-2, q_caf=0.002, q_tf=0.2, p0_caf=0.03, p0_tf=3.0,
    jump_thr=12.0, jump_caf=0.03, jump_tf=2.0,
    # MPC: horizon, move blocks (v1 for m1 samples, v2 for m2, then steady-state jacket), grid
    N=20, m1=2, m2=4, ngrid=15, lam=0.02, rho=0.0, t_soft=331.0, w_t=50.0,
    umin=295.0, umax=302.0,
    # hard safety override on the measured reactor temperature (runaway limit is 335 K)
    t_trip=332.0,
)


def deriv(Ca, T, u, Caf, Tf, P):
    k = P['kr'] * np.exp(-P['ER'] * (1.0 / T - 1.0 / P['Tref']))
    r = k * Ca
    return P['a'] * (Caf - Ca) - r, P['a'] * (Tf - T) + P['b'] * r + P['c'] * (u - T)


def step(Ca, T, u, Caf, Tf, P, n=2):
    h = DT / n
    for _ in range(n):
        a1, b1 = deriv(Ca, T, u, Caf, Tf, P)
        a2, b2 = deriv(Ca + 0.5 * h * a1, T + 0.5 * h * b1, u, Caf, Tf, P)
        a3, b3 = deriv(Ca + 0.5 * h * a2, T + 0.5 * h * b2, u, Caf, Tf, P)
        a4, b4 = deriv(Ca + h * a3, T + h * b3, u, Caf, Tf, P)
        Ca = Ca + h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        T = T + h / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
        # keep predictions in a physical box so a wild candidate cannot overflow
        Ca = np.clip(Ca, 0.0, 2.0)
        T = np.clip(T, 250.0, 450.0)
    return Ca, T


class Controller:
    def __init__(self, params=None):
        P = dict(DEFAULTS)
        if params:
            P.update(params)
        self.P = P
        self.x = None
        self.u_prev = None
        self.R = np.diag([P['r_ca'] ** 2, P['r_t'] ** 2])
        self.Q = np.diag([P['q_ca'] ** 2, P['q_t'] ** 2, P['q_caf'] ** 2, P['q_tf'] ** 2])
        self.eps = np.array([1e-5, 1e-3, 1e-5, 1e-3])

    # ---------- estimator ----------
    def _reset(self, y):
        P = self.P
        self.x = np.array([y[0], y[1], P['Caf'], P['Tf']])
        self.Pc = np.diag([P['r_ca'] ** 2, P['r_t'] ** 2, P['p0_caf'] ** 2, P['p0_tf'] ** 2])

    def _predict(self, u):
        P = self.P
        X = np.tile(self.x, (9, 1))
        for i in range(4):
            X[1 + 2 * i, i] += self.eps[i]
            X[2 + 2 * i, i] -= self.eps[i]
        Ca, T = step(X[:, 0], X[:, 1], u, X[:, 2], X[:, 3], P)
        Y = X.copy()
        Y[:, 0] = Ca
        Y[:, 1] = T
        F = np.zeros((4, 4))
        for i in range(4):
            F[:, i] = (Y[1 + 2 * i] - Y[2 + 2 * i]) / (2 * self.eps[i])
        self.x = Y[0].copy()
        self.Pc = F @ self.Pc @ F.T + self.Q

    def _update(self, y):
        P = self.P
        Pc = self.Pc
        S = Pc[:2, :2] + self.R
        inn = y - self.x[:2]
        nis = float(inn @ np.linalg.solve(S, inn))
        if nis > P['jump_thr']:
            # innovation too large for a quiet plant: a feed shift is likely, reopen the feed states
            Pc = Pc.copy()
            Pc[2, 2] += P['jump_caf'] ** 2
            Pc[3, 3] += P['jump_tf'] ** 2
            S = Pc[:2, :2] + self.R
        K = Pc[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ inn
        self.Pc = Pc - K @ Pc[:2, :]
        self.Pc = 0.5 * (self.Pc + self.Pc.T)
        self.x[2] = min(max(self.x[2], 0.85), 1.25)
        self.x[3] = min(max(self.x[3], P['Tf'] - 25.0), P['Tf'] + 25.0)

    # ---------- steady state ----------
    def u_ss(self, sp, Caf, Tf):
        P = self.P
        sp = min(max(sp, 0.05), Caf - 1e-3)
        k = P['a'] * (Caf - sp) / sp
        Tss = 1.0 / (1.0 / P['Tref'] - math.log(k / P['kr']) / P['ER'])
        u = Tss - (P['a'] * (Tf - Tss) + P['b'] * k * sp) / P['c']
        if not math.isfinite(u):
            u = 0.5 * (P['umin'] + P['umax'])
        return min(max(u, P['umin']), P['umax'])

    # ---------- MPC ----------
    def _cost(self, v1, v2, uss, sp):
        P = self.P
        Ca = np.full(v1.shape, self.x[0])
        T = np.full(v1.shape, self.x[1])
        Caf, Tf = self.x[2], self.x[3]
        J = np.zeros(v1.shape)
        for j in range(P['N']):
            u = v1 if j < P['m1'] else (v2 if j < P['m1'] + P['m2'] else uss)
            Ca, T = step(Ca, T, u, Caf, Tf, P, n=1)
            J += ((Ca - sp) / 0.01) ** 2 + P['w_t'] * np.maximum(T - P['t_soft'], 0.0) ** 2
        J += P['lam'] * (v1 - self.u_prev) ** 2 + P['rho'] * (v2 - v1) ** 2
        return np.where(np.isfinite(J), J, 1e30)

    def _mpc(self, sp):
        P = self.P
        uss = self.u_ss(sp, self.x[2], self.x[3])
        g = np.linspace(P['umin'], P['umax'], P['ngrid'])
        V1, V2 = np.meshgrid(g, g, indexing='ij')
        v1 = np.append(V1.ravel(), [uss, self.u_prev])
        v2 = np.append(V2.ravel(), [uss, uss])
        J = self._cost(v1, v2, uss, sp)
        i = int(np.argmin(J))
        b1, b2 = v1[i], v2[i]
        w = (P['umax'] - P['umin']) / (P['ngrid'] - 1)
        for _ in range(3):
            o = np.linspace(-w, w, 9)
            O1, O2 = np.meshgrid(o, o, indexing='ij')
            c1 = np.clip(b1 + O1.ravel(), P['umin'], P['umax'])
            c2 = np.clip(b2 + O2.ravel(), P['umin'], P['umax'])
            J = self._cost(c1, c2, uss, sp)
            i = int(np.argmin(J))
            b1, b2 = c1[i], c2[i]
            w /= 4.0
        return float(b1)

    def act(self, obs):
        P = self.P
        y = np.array([float(obs['Ca']), float(obs['T'])])
        sp = float(obs['Ca_sp'])
        if not np.all(np.isfinite(y)):
            # bad measurement: hold the last move (or mid-range at start-up)
            return self.u_prev if self.u_prev is not None else 0.5 * (P['umin'] + P['umax'])
        if self.x is None:
            self._reset(y)
            self.u_prev = self.u_ss(sp, P['Caf'], P['Tf'])
        else:
            self._predict(self.u_prev)
            self._update(y)
            if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.Pc)):
                self._reset(y)
        try:
            u = self._mpc(sp)
        except Exception:
            u = self.u_ss(sp, self.x[2], self.x[3])
        if not math.isfinite(u):
            u = self.u_ss(sp, self.x[2], self.x[3])
        if y[1] > P['t_trip']:
            u = P['umin']  # safety: full cooling if the reactor approaches the runaway limit
        u = min(max(u, P['umin']), P['umax'])
        self.u_prev = u
        return u
