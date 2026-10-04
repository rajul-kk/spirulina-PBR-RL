"""Distribution of EKF feed estimates (Caf, Tf) over all batches for a given parameter set."""
import numpy as np, sys, json
from an_common import load
import sim
mod = sim.load_ctrl(sys.argv[1]); prm = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
caf = []; tf = []; first = []
for name, d in load('*').items():
    c = mod.Controller(prm); est = []
    for k in range(len(d)):
        c.act(dict(t_min=d['t_min'][k], Ca=d['Ca'][k], T=d['T'][k], Ca_sp=d['Ca_sp'][k])); c.u_prev = d['Tc'][k]; est.append(c.x.copy())
    est = np.array(est); caf += list(est[15:, 2]); tf += list(est[15:, 3]); first.append(est[15, 2:])
caf = np.array(caf); tf = np.array(tf); first = np.array(first)
print('Caf mean %.4f sd %.4f p5 %.3f p95 %.3f' % (caf.mean(), caf.std(), *np.percentile(caf, [5, 95])))
print('Tf  mean %.2f sd %.2f p5 %.1f p95 %.1f' % (tf.mean(), tf.std(), *np.percentile(tf, [5, 95])))
print('at k=15: Caf mean %.4f sd %.4f ; Tf mean %.2f sd %.2f' % (first[:, 0].mean(), first[:, 0].std(), first[:, 1].mean(), first[:, 1].std()))
