"""Same start-up / setpoint-change / steady split as breakdown.py, but on my simulator (true Ca known)."""
import sys, json, numpy as np, simlab
from controller import Controller
params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
ns = int(sys.argv[2]) if len(sys.argv) > 2 else 60
tot = []
for s in range(300, 300+ns):
    c, tm, tr = simlab.run_batch(Controller, params, s, trace=True)
    sp = tr[:, 3]; N = len(sp); e2 = ((tr[1:, 1]-sp[:-1])/0.01)**2
    ch = np.nonzero(np.diff(sp))[0]+1; lab = np.zeros(N-1, int); lab[:12] = 1
    for k in ch: lab[k:k+12] = 2
    # steady split: within 12 samples after a disturbance step or not
    dstep = np.nonzero((np.diff(tr[:, 5]) != 0) | (np.diff(tr[:, 6]) != 0))[0]+1
    for k in dstep: lab[k:k+12] = np.where(lab[k:k+12] == 0, 3, lab[k:k+12])
    tot.append([c] + [e2[lab == i].sum()/N for i in range(4)])
tot = np.array(tot)
print("SIM %s: total %.3f (median %.3f) = quiet %.4f + startup %.3f + spchange %.3f + after-disturbance %.4f" % (params, tot[:, 0].mean(), np.median(tot[:, 0]), *tot[:, 1:].mean(0)))
