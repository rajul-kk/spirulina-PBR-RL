"""Stress test: controller (refit model) on perturbed surrogate plants; reports cost and max T."""
import numpy as np, json, sys, sim
from sim_cross import REFIT
mod = sim.load_ctrl(sys.argv[1]); n = int(sys.argv[2]); base = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
ctrlp = dict(base); pass
cases = {'nominal': {}, 'kr+25%': dict(kr=REFIT['kr'] * 1.25), 'kr-25%': dict(kr=REFIT['kr'] * 0.75),
         'b+15%': dict(b=REFIT['b'] * 1.15), 'c-15%': dict(c=REFIT['c'] * 0.85), 'a+15%': dict(a=REFIT['a'] * 1.15),
         'ER+5%': dict(ER=REFIT['ER'] * 1.05)}
for name, ch in cases.items():
    plant = dict(REFIT); plant.update(ch); rng = np.random.default_rng(1); c = []; tm = []
    for _ in range(n):
        j, rows = sim.run(mod, sim.scenario(rng, plant), ctrlp, true=plant, log=True); c.append(j); tm.append(rows[:, 2].max())
    print('%-8s mean %.3f med %.3f max %.2f | maxT %.1f' % (name, np.mean(c), np.median(c), np.max(c), np.max(tm)), flush=True)
