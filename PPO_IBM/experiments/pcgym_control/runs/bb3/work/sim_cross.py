"""Model-mismatch check: surrogate plant built from one parameter set, controller using another."""
import numpy as np, json, sys, sim
REFIT = dict(a=0.73716, kr=0.11696, ER=8775.66, b=190.579, c=2.12598, Caf=1.0357, Tf=365.37, Tref=323.0)
ORIG = dict(sim.TRUE)
if __name__ == '__main__':
    mod = sim.load_ctrl(sys.argv[1]); n = int(sys.argv[2]); base = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    for pn, plant in (('orig', ORIG), ('refit', REFIT)):
        for cn, cp in (('orig', ORIG), ('refit', REFIT)):
            p = dict(base); p.update({k: v for k, v in cp.items() if k != 'Tref'})
            c = sim.evaluate(sys.argv[1], p, n=n, true=plant)
            print('plant %-5s ctrl %-5s mean %.4f med %.3f max %.2f' % (pn, cn, c.mean(), np.median(c), c.max()), flush=True)
