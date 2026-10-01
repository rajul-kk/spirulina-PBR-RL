import numpy as np, plantsim
C = plantsim.load("controller.py")
sc = plantsim.scenario(200005)
c = C()
ca0, T0 = sc["x0"]; ti, caf, sp = sc["Ti"][0], sc["Caf"][0], sc["sp"][0]
print(ca0, T0, ti, caf, sp)
import controller as cm
u_tail = min(max(cm.steady_tc(sp, ti, caf), 295), 302)
M = c.M
v = np.full(M, u_tail)
V = np.tile(v, (M + 1, 1)); V[1:] += 0.02 * np.eye(M)
Rr = c._resid(V, u_tail, ca0, T0, ti, caf, sp)
r = Rr[0]; J = (Rr[1:] - r).T / 0.02
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print("r", r[:25]); print("J col0", J[:25, 0]); print("g", J.T @ r)
print("GN unconstrained", np.linalg.solve(J.T @ J + 1e-6 * np.eye(M), -J.T @ r))
print("u", c._mpc(ca0, T0, ti, caf, sp), c.plan)
