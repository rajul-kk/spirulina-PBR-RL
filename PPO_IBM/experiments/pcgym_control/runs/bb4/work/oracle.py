"""Headroom check in simulation: MPC fed the true state and true feed (no estimator)."""
import sys, numpy as np, model, simtest
from controller import Controller
def run_oracle(params, sc, rng):
    c = Controller(params); Ca, T = sc['x0']; cost = 0.0; ph = np.zeros(120)
    for k in range(120):
        rng.normal(); rng.normal()
        c.x = [Ca, T, sc['Caf'][k], sc['Tf'][k]]
        u = c._mpc(c.x, sc['sp'][k], c.u_prev if c.u_prev is not None else 298.5); u = min(max(u, 295.0), 302.0); c.u_prev = u
        ph[k] = ((Ca - sc['sp'][k]) / 0.01) ** 2
        Ca, T = model.step(Ca, T, u, sc['Caf'][k], sc['Tf'][k], model.P_FIT)
    return ph
params = eval(sys.argv[2]) if len(sys.argv) > 2 else None
tot = []; first = []
for i in range(int(sys.argv[1])):
    rng = np.random.default_rng(i); sc = simtest.scenario(rng); ph = run_oracle(params, sc, rng); sp = sc['sp']
    ch = [0] + [j for j in range(1, 120) if sp[j] != sp[j - 1]]
    tot.append(ph.mean()); first.append(sum(ph[j] for j in ch) / 120)
print('oracle mean %.3f median %.3f ; of which the unavoidable first sample after each change %.3f' % (np.mean(tot), np.median(tot), np.mean(first)))
