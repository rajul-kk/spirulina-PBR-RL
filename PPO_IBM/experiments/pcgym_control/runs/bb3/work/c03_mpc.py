import math
import numpy as np

DT = 13.0 / 60.0  # min per sample

DEFAULTS = dict(
    # identified grey-box model: dCa/dt = a(Caf-Ca) - k Ca ; dT/dt = a(Tf-T) + b k Ca + c(Tc-T)
    a=0.834, kr=0.1048, ER=9183.0, b=216.3, c=2.033, Caf=0.9948, Tf=357.6, Tref=323.0,
    # estimator
    r_ca=0.002, r_t=0.2, q_ca=1e-4, q_t=1e-2, q_caf=0.004, q_tf=0.4, p0_caf=0.03, p0_tf=3.0,
    jump_thr=1e9, jump_caf=0.02, jump_tf=1.5,
    # MPC
    N=20, m1=2, m2=4, ngrid=15, lam=0.02, rho=0.0, t_soft=331.0, w_t=50.0,
    umin=295.0, umax=302.0,
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
    def _predict(self, u):
        P = self.P
        x = self.x
        X = np.tile(x, (9, 1))
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
        Pc = self.Pc
        S = Pc[:2, :2] + self.R
        K = Pc[:, :2] @ np.linalg.inv(S)
        inn = y - self.x[:2]
        nis = float(inn @ np.linalg.solve(S, inn))
        if nis > self.P['jump_thr']:
            # innovation too large for the quiet-plant hypothesis: a feed shift is likely, reopen the disturbance states
            Pc = Pc.copy()
            Pc[2, 2] += self.P['jump_caf'] ** 2
            Pc[3, 3] += self.P['jump_tf'] ** 2
            S = Pc[:2, :2] + self.R
            K = Pc[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ inn
        self.Pc = Pc - K @ Pc[:2, :]
        self.Pc = 0.5 * (self.Pc + self.Pc.T)
        self.x[2] = min(max(self.x[2], 0.85), 1.15)
        return inn, S

    # ---------- steady state ----------
    def u_ss(self, sp, Caf, Tf):
        P = self.P
        sp = min(sp, Caf - 1e-3)
        k = P['a'] * (Caf - sp) / sp
        Tss = 1.0 / (1.0 / P['Tref'] - math.log(k / P['kr']) / P['ER'])
        u = Tss - (P['a'] * (Tf - Tss) + P['b'] * k * sp) / P['c']
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
        return J

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
        if self.x is None:
            self.x = np.array([y[0], y[1], P['Caf'], P['Tf']])
            self.Pc = np.diag([P['r_ca'] ** 2, P['r_t'] ** 2, P['p0_caf'] ** 2, P['p0_tf'] ** 2])
            self.u_prev = self.u_ss(sp, P['Caf'], P['Tf'])
        else:
            self._predict(self.u_prev)
            self._update(y)
        u = self._mpc(sp)
        u = min(max(u, P['umin']), P['umax'])
        self.u_prev = u
        return u
