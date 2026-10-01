"""Model adequacy on batches NOT used in the fit: innovation mean, std, lag-1 autocorrelation,
and 5-step-ahead open-loop prediction error, for fit2 and fit3 parameter sets."""
import sys, glob
import numpy as np
sys.path.insert(0, "runs/bb2/work")
from show import load
import fit
files = [p for pat in sys.argv[1:] for p in sorted(glob.glob("runs/bb2/trials/" + pat))]
B = [load(p) for p in files]
for name in ("fit2_p.npy", "fit3_p.npy"):
    p = np.load("runs/bb2/work/" + name)
    th, sCa, sT, qCaf, qTf, x0d = fit.unpack(p)
    E = []; nll = 0; e5 = []
    for d in B:
        v, xs, inn = fit.ekf(th, d, sCa, sT, qCaf, qTf, x0d)
        nll += v; E.append(inn[10:])
        for i in range(10, 114, 3):
            Ca, T = xs[i][0], xs[i][1]
            for j in range(5):
                Ca, T = fit.step(Ca, T, xs[i][2], xs[i][3], d["Tc"][i + j], th)
            e5.append((d["Ca"][i + 5] - Ca, d["T"][i + 5] - T))
    E = np.concatenate(E); e5 = np.array(e5)
    ac = [np.corrcoef(E[:-1, k], E[1:, k])[0, 1] for k in (0, 1)]
    print("%s nll %.1f | innov mean Ca %+.5f T %+.4f | std Ca %.5f T %.4f | lag1 corr Ca %+.3f T %+.3f | 5-step err std Ca %.5f T %.4f" % (
        name, nll, E[:, 0].mean(), E[:, 1].mean(), E[:, 0].std(), E[:, 1].std(), ac[0], ac[1], e5[:, 0].std(), e5[:, 1].std()))
