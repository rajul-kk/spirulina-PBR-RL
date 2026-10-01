import numpy as np, plantsim
import controller as cm
c = cm.Controller()
ca0, T0, ti, caf, sp = 0.8591, 329.76, 351.18, 1.0127, 0.8755
u_tail = min(max(cm.steady_tc(sp, ti, caf), 295), 302)
print("u_tail", cm.steady_tc(sp, ti, caf))
M = c.M
np.set_printoptions(precision=4, suppress=True, linewidth=200)
for v0 in (302.0, 300.0):
    v = np.full(M, v0)
    V = np.tile(v, (M + 1, 1)); V[1:] += 0.02 * np.eye(M)
    Rr = c._resid(V, u_tail, ca0, T0, ti, caf, sp)
    r = Rr[0]; J = (Rr[1:] - r).T / 0.02
    print("r", r[:25]); print("J col0", J[:25, 0]); print("g", J.T @ r)
    Hm = J.T @ J + 1e-6 * np.eye(M)
    print("d", cm._box_qp(Hm, J.T @ r, 295 - v, 302 - v))
