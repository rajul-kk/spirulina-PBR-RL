"""Output-error (simulation) fit: global kinetics/heat params; per batch initial state and
piecewise-constant feed concentration / feed temperature with one changepoint each."""
import sys, numpy as np
from scipy.optimize import least_squares
from fit1 import f, DT, load
SC, ST = 0.002, 0.2
def sim(p, d, Ca0, T0, Caf, Tf, n=3):
    N = len(d['Tc']); Ca = np.empty(N); T = np.empty(N); Ca[0] = Ca0; T[0] = T0
    h = DT / n; ca, t = Ca0, T0
    for i in range(N - 1):
        u = d['Tc'][i]; cf = Caf[i]; tf = Tf[i]
        for _ in range(n):
            k1 = f(ca, t, u, p, cf, tf)
            k2 = f(ca + h/2*k1[0], t + h/2*k1[1], u, p, cf, tf)
            k3 = f(ca + h/2*k2[0], t + h/2*k2[1], u, p, cf, tf)
            k4 = f(ca + h*k3[0], t + h*k3[1], u, p, cf, tf)
            ca += h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); t += h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        Ca[i+1] = ca; T[i+1] = t
    return Ca, T
def unpack(theta, D, cps):
    p = theta[:5]; out = []
    for j, (fn, d) in enumerate(D):
        b = theta[5 + 6*j: 11 + 6*j]; N = len(d['Tc']); i = np.arange(N)
        Caf = np.where(i < cps[j][0], b[2], b[3]); Tf = np.where(i < cps[j][1], b[4], b[5])
        out.append((b[0], b[1], Caf, Tf))
    return p, out
def resid(theta, D, cps, sep=False):
    p, per = unpack(theta, D, cps); out = []
    for (fn, d), (c0, t0, Caf, Tf) in zip(D, per):
        Ca, T = sim(p, d, c0, t0, Caf, Tf)
        e = ((d['Ca'] - Ca) / SC, (d['T'] - T) / ST)
        out.append(e if sep else np.concatenate(e))
    return out if sep else np.concatenate(out)
def fit(D, cps, th0=None, pfix=None):
    if th0 is None:
        th0 = [1.0, np.log(0.1), 8500.0, 200.0, 2.0]
        for fn, d in D: th0 += [d['Ca'][0], d['T'][0], 1.0, 1.0, 350.0, 350.0]
    xs = [1, 1, 1000, 100, 1] + [0.1, 10, 0.1, 0.1, 10, 10] * len(D)
    return least_squares(resid, np.array(th0, float), args=(D, cps), x_scale=xs)
if __name__ == '__main__':
    D = load(sys.argv[1:]); N = 120
    cps = [(60, 60)] * len(D); best = None
    # coordinate search over changepoints
    r = fit(D, cps); th = r.x
    for it in range(3):
        for j in range(len(D)):
            for which in (0, 1):
                cands = list(range(5, 116, 5)); costs = []
                for c in cands:
                    cp = list(cps); t = list(cp[j]); t[which] = c; cp[j] = tuple(t)
                    # quick: refit only this batch's 6 params with globals fixed
                    def rj(b, cp=cp):
                        th2 = th.copy(); th2[5+6*j:11+6*j] = b
                        p, per = unpack(th2, D, cp); c0, t0, Caf, Tf = per[j]; d = D[j][1]
                        Ca, T = sim(p, d, c0, t0, Caf, Tf)
                        return np.concatenate([(d['Ca'] - Ca) / SC, (d['T'] - T) / ST])
                    rr = least_squares(rj, th[5+6*j:11+6*j], x_scale=[0.1, 10, 0.1, 0.1, 10, 10])
                    costs.append((rr.cost, c, rr.x))
                # refine +-4 around the best
                c0 = min(costs)[1]
                for c in range(max(2, c0 - 4), min(118, c0 + 5)):
                    cp = list(cps); t = list(cp[j]); t[which] = c; cp[j] = tuple(t)
                    def rj(b, cp=cp):
                        th2 = th.copy(); th2[5+6*j:11+6*j] = b
                        p, per = unpack(th2, D, cp); c0_, t0, Caf, Tf = per[j]; d = D[j][1]
                        Ca, T = sim(p, d, c0_, t0, Caf, Tf)
                        return np.concatenate([(d['Ca'] - Ca) / SC, (d['T'] - T) / ST])
                    rr = least_squares(rj, th[5+6*j:11+6*j], x_scale=[0.1, 10, 0.1, 0.1, 10, 10])
                    costs.append((rr.cost, c, rr.x))
                cst, c, xb = min(costs, key=lambda z: z[0])
                t = list(cps[j]); t[which] = c; cps[j] = tuple(t); th[5+6*j:11+6*j] = xb
        r = fit(D, cps, th); th = r.x
        print('iter', it, 'cost', r.cost, 'cps', cps)
        print(' a lnk E B c =', np.round(th[:5], 4), 'kref', np.exp(th[1]))
    J = r.jac; cov = np.linalg.pinv(J.T @ J); print('sd globals', np.round(np.sqrt(np.diag(cov))[:5], 4))
    for j, ((fn, d), (eC, eT)) in enumerate(zip(D, resid(th, D, cps, sep=True))):
        b = th[5+6*j:11+6*j]
        print(fn, 'cp', cps[j], 'Caf %.4f->%.4f Tf %.2f->%.2f' % tuple(b[2:]), 'rms eCa %.4f eT %.3f' % (np.std(eC)*SC, np.std(eT)*ST))
    np.save('runs/bb4/work/fit2_theta.npy', th)
