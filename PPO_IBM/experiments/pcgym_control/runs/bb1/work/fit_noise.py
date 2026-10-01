"""Tune the EKF noise model on logged closed-loop plant data by maximum likelihood of the innovations
(no plant calls). Parameters: r_ca, r_t, q_ca, q_t, q_caf, q_tf (log-scale)."""
import glob, sys, numpy as np
from scipy.optimize import minimize
from controller import Controller, DEFAULTS
files = sorted(glob.glob("../trials/b*_final.csv"))
fit_files, test_files = files[0::2][:35], files[1::2]
load = lambda fs: [np.genfromtxt(f, delimiter=",", names=True) for f in fs]
Dfit, Dtest = load(fit_files), load(test_files)
names = ["r_ca", "r_t", "q_ca", "q_t", "q_caf", "q_tf"]
def nll(logp, D):
    prm = dict(zip(names, np.exp(logp))); tot = 0.0; n = 0; acs = []
    for d in D:
        c = Controller(prm); inn = []
        c.x = np.array([d["Ca"][0], d["T"][0], c.p["caf0"], c.p["tf0"]]); c.P = np.diag([c.p["r_ca"]**2, c.p["r_t"]**2, c.p["p0_caf"]**2, c.p["p0_tf"]**2])
        for k in range(1, len(d)):
            c._predict(d["Tc"][k-1]); v = np.array([d["Ca"][k]-c.x[0], d["T"][k]-c.x[1]]); S = c.P[:2, :2] + c.R
            tot += 0.5*(v @ np.linalg.solve(S, v) + np.log(np.linalg.det(S))); n += 1; inn.append(v)
            c._update(d["Ca"][k], d["T"][k])
        inn = np.array(inn); acs.append([np.corrcoef(inn[10:-1, j], inn[11:, j])[0, 1] for j in (0, 1)])
    return tot/n, np.mean(acs, 0)
x0 = np.log([DEFAULTS[k] for k in names])
print("default: fit nll %.4f test nll %.4f autocorr %s" % (nll(x0, Dfit)[0], *nll(x0, Dtest)), flush=True)
it = [0]
def f(x):
    v = nll(x, Dfit)[0]; it[0] += 1
    if it[0] % 20 == 0: print(it[0], round(v, 4), dict(zip(names, np.exp(x).round(5))), flush=True)
    return v
r = minimize(f, x0, method="Nelder-Mead", options=dict(maxfev=220, xatol=0.03, fatol=1e-4, initial_simplex=np.vstack([x0] + [x0 + 0.5*np.eye(6)[i] for i in range(6)])))
print("ML:", dict(zip(names, np.exp(r.x).round(6))))
print("ML: fit nll %.4f test nll %.4f autocorr %s" % (r.fun, *nll(r.x, Dtest)), flush=True)
