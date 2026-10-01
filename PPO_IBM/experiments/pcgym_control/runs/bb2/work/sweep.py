"""Offline tuning sweeps on my own fitted-model simulator (no plant batches used)."""
import sys, ast
import numpy as np
sys.path.insert(0, "runs/bb2/work")
import sim

FIT = dict(a=1.0445, kref=0.11245, ER=8948.0, b=207.54, c=2.0758, Caf0=1.0, Tf0=349.0)

if __name__ == "__main__":
    path = sys.argv[1]
    n = int(sys.argv[2])
    Ctrl = sim.load_ctrl(path)
    for spec in sys.argv[3:]:
        prm = dict(FIT)
        prm.update(ast.literal_eval(spec))
        th = prm.pop("_th", sim.TH)
        m, md, mx, tm = sim.evaluate(Ctrl, n, params=prm, th=th)
        print("%-70s mean %.4f median %.4f max %.3f Tmax %.1f" % (spec, m, md, mx, tm), flush=True)
