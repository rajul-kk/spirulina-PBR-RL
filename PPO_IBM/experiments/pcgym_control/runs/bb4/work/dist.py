"""Disturbance statistics: per-sample implied feed concentration / temperature from one-step
model residuals, then piecewise-constant segmentation (0-3 changepoints, penalised)."""
import sys, glob, itertools, numpy as np
import model
P = dict(model.P_FIT)

def implied(d, p=P):
    N = len(d['Ca']); caf = np.empty(N - 1); tf = np.empty(N - 1)
    for i in range(N - 1):
        c0, t0 = model.step(d['Ca'][i], d['T'][i], d['Tc'][i], 1.0, 350.0, p)
        c1, t1 = model.step(d['Ca'][i], d['T'][i], d['Tc'][i], 1.01, 350.0, p)
        c2, t2 = model.step(d['Ca'][i], d['T'][i], d['Tc'][i], 1.0, 351.0, p)
        # 2x2 linear solve for (dCaf, dTf)
        A = np.array([[(c1 - c0) / 0.01, c2 - c0], [(t1 - t0) / 0.01, t2 - t0]])
        s = np.linalg.solve(A, [d['Ca'][i + 1] - c0, d['T'][i + 1] - t0])
        caf[i] = 1.0 + s[0]; tf[i] = 350.0 + s[1]
    return caf, tf

def segment(y, pen, maxk=3, minlen=3):
    N = len(y); cs = np.r_[0, np.cumsum(y)]; cs2 = np.r_[0, np.cumsum(y * y)]
    def sse(a, b): return cs2[b] - cs2[a] - (cs[b] - cs[a]) ** 2 / (b - a)
    best = (sse(0, N), [])
    for k in range(1, maxk + 1):
        bk = None
        for cps in itertools.combinations(range(minlen, N - minlen + 1, 1 if k < 3 else 3), k):
            if any(b - a < minlen for a, b in zip((0,) + cps, cps + (N,))): continue
            s = sum(sse(a, b) for a, b in zip((0,) + cps, cps + (N,)))
            if bk is None or s < bk[0]: bk = (s, list(cps))
        if bk and bk[0] + pen * k < best[0] + pen * len(best[1]): best = bk
    cps = best[1]; means = [y[a:b].mean() for a, b in zip([0] + cps, cps + [N])]
    return cps, means

if __name__ == '__main__':
    allc = []; allt = []
    for pat in sys.argv[1:]:
        for f in sorted(glob.glob('runs/bb4/trials/' + pat)):
            d = np.genfromtxt(f, delimiter=',', names=True)
            caf, tf = implied(d)
            sc = np.std(np.diff(caf)) / 1.414; st = np.std(np.diff(tf)) / 1.414
            c_cp, c_m = segment(caf, pen=12 * sc ** 2); t_cp, t_m = segment(tf, pen=12 * st ** 2)
            print(f[-14:], 'Caf', c_cp, np.round(c_m, 4), '| Tf', t_cp, np.round(t_m, 2), '| noise %.4f %.2f' % (sc, st))
            allc.append((c_cp, c_m)); allt.append((t_cp, t_m))
    for name, al in (('Caf', allc), ('Tf', allt)):
        init = [m[0] for _, m in al]; jumps = [m[i + 1] - m[i] for _, m in al for i in range(len(m) - 1)]
        times = [c for cp, _ in al for c in cp]; ncp = [len(cp) for cp, _ in al]
        print(name, 'initial mean %.4f sd %.4f range %.4f..%.4f' % (np.mean(init), np.std(init), min(init), max(init)))
        print('   n changes per batch', np.bincount(ncp), 'jump sizes', np.round(sorted(jumps), 4))
        print('   change times', sorted(times))
