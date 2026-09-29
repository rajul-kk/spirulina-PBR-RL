import numpy as np, sys, glob, os
from calib import analyse
import contextlib, io
bins = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3, 10]
allo = []
for f in sorted(glob.glob("../trials/b*.csv")):
    b = os.path.basename(f)[:-4]
    if any(k in b for k in sys.argv[1:]) or not sys.argv[1:]:
        with contextlib.redirect_stdout(io.StringIO()):
            o = analyse(b, show=False)
        # per-batch strain factor: ratio at low od (<0.6) if available
        m = np.isfinite(o[:, 7])
        allo.append((b, o[m]))
for b, o in allo:
    line = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (o[:, 1] >= lo) & (o[:, 1] < hi)
        if s.sum() >= 3:
            line.append(f"[{lo}-{hi}) {o[s,7].mean()/o[s,6].mean():.2f} C{o[s,2].mean():.1f}")
    print(b, " ".join(line))
