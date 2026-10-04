"""Multi-step open-loop prediction check at setpoint changes and start-up: model from EKF state vs measured.
usage: an07_multistep.py ctrl pattern [json params]"""
import numpy as np, sys, json
from an_common import load
import sim
mod = sim.load_ctrl(sys.argv[1]); prm = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
H = 10; errs = {'up': [], 'down': [], 'start': []}; errT = {'up': [], 'down': [], 'start': []}
for name, d in load(sys.argv[2]).items():
    c = mod.Controller(prm); ch = list(np.nonzero(np.diff(d['Ca_sp']))[0] + 1)
    for k in range(len(d)):
        c.act(dict(t_min=d['t_min'][k], Ca=d['Ca'][k], T=d['T'][k], Ca_sp=d['Ca_sp'][k])); c.u_prev = d['Tc'][k]
        tag = None
        if k == 2: tag = 'start'
        elif k in ch and k + H < 120 and abs(d['Ca_sp'][k] - d['Ca_sp'][k - 1]) > 0.015:
            tag = 'up' if d['Ca_sp'][k] > d['Ca_sp'][k - 1] else 'down'
        if tag:
            Ca, T = c.x[0], c.x[1]; e = []; eT = []
            for j in range(H):
                Ca, T = mod.step(Ca, T, d['Tc'][k + j], c.x[2], c.x[3], c.P)
                e.append(Ca - d['Ca'][k + j + 1]); eT.append(T - d['T'][k + j + 1])
            errs[tag].append(e); errT[tag].append(eT)
for t in errs:
    e = np.array(errs[t]); eT = np.array(errT[t])
    print(t, 'n', len(e)); print('  Ca bias x1e3', np.round(e.mean(0) * 1e3, 1)); print('  Ca sd   x1e3', np.round(e.std(0) * 1e3, 1))
    print('  T bias', np.round(eT.mean(0), 2)); print('  T sd  ', np.round(eT.std(0), 2))
