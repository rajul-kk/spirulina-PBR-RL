"""Replay a logged batch through a controller's estimator: usage an06_replay.py ctrl batch k0 k1 [json params]"""
import numpy as np, sys, json
from an_common import load
import sim
mod = sim.load_ctrl(sys.argv[1]); d = list(load(sys.argv[2].split('_', 1)[1] if False else '*').items())
d = dict(d)[sys.argv[2]]; k0, k1 = int(sys.argv[3]), int(sys.argv[4])
c = mod.Controller(json.loads(sys.argv[5]) if len(sys.argv) > 5 else None)
for k in range(len(d)):
    obs = dict(t_min=d['t_min'][k], Ca=d['Ca'][k], T=d['T'][k], Ca_sp=d['Ca_sp'][k])
    u = c.act(obs); c.u_prev = d['Tc'][k]
    if k0 <= k < k1:
        print('%3d Ca %.4f T %.2f sp %.4f u_log %.2f u_now %.2f | est %.4f %.2f Caf %.4f Tf %.2f' % (k, d['Ca'][k], d['T'][k], d['Ca_sp'][k], d['Tc'][k], u, *c.x))
