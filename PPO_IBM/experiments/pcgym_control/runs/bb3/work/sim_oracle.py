"""Surrogate floor: same MPC but fed the true state and feed conditions (no estimation error)."""
import numpy as np, sys, json, sim
mod = sim.load_ctrl(sys.argv[1]); n = int(sys.argv[2]); prm = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
rng = np.random.default_rng(0); out = []; base = []; dus = []
for i in range(n):
    sc = sim.scenario(rng)
    c = mod.Controller(prm); Ca, T = sc['Ca0'], sc['T0']; J = 0
    c.u_prev = 298.5
    for k in range(120):
        c.x = np.array([Ca, T, sc['Caf'][k], sc['Tf'][k]])
        u = c._mpc(sc['sp'][k]); c.u_prev = u
        J += ((Ca - sc['sp'][k]) / 0.01) ** 2
        Ca, T = sim._step(Ca, T, u, sc['Caf'][k], sc['Tf'][k], sim.TRUE, n=4)
    out.append(J / 120)
    j, rows = sim.run(mod, sc, prm, log=True); base.append(j); dus.append(np.std(np.diff(rows[:, 4])))
print('oracle mean %.3f med %.3f | estimator-based mean %.3f med %.3f | sd(dTc) %.2f' % (np.mean(out), np.median(out), np.mean(base), np.median(base), np.mean(dus)))
