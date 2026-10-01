"""Compare quiet-phase behaviour plant vs my simulator: Tc move std, measured Ca-sp std, innovation std."""
import glob, numpy as np, simlab
from controller import Controller
def stats(Ca_m, T_m, sp, Tc):
    N = len(sp); ch = np.nonzero(np.diff(sp))[0]+1; q = np.ones(N, bool); q[:15] = False
    for k in ch: q[k:k+15] = False
    c = Controller(); inn = np.zeros((N, 2))
    for k in range(N):
        if k == 0:
            c.x = np.array([Ca_m[0], T_m[0], c.p["caf0"], c.p["tf0"]]); c.P = np.diag([c.p["r_ca"]**2, c.p["r_t"]**2, c.p["p0_caf"]**2, c.p["p0_tf"]**2])
        else:
            c._predict(Tc[k-1]); inn[k] = (Ca_m[k]-c.x[0], T_m[k]-c.x[1]); c._update(Ca_m[k], T_m[k])
    q1 = q[1:] & q[:-1]
    return [np.diff(Tc)[q1].std(), np.sqrt(np.mean((Ca_m-sp)[q]**2)), np.median(np.abs(inn[q, 0]))*1.4826, np.median(np.abs(inn[q, 1]))*1.4826, inn[q, 0].std(), inn[q, 1].std()]
P = []
for f in sorted(glob.glob("../trials/b*_final.csv")):
    d = np.genfromtxt(f, delimiter=",", names=True); P.append(stats(d["Ca"], d["T"], d["Ca_sp"], d["Tc"]))
S = []
for s in range(400, 440):
    rng = np.random.default_rng(s); sp, caf, tf, x = simlab.scenario(rng); c = Controller(); L = []
    for k in range(simlab.N):
        cm = x[0]+rng.normal(0, .0018); tm = x[1]+rng.normal(0, .19); u = float(np.clip(c.act(dict(t_min=0, Ca=cm, T=tm, Ca_sp=sp[k])), 295, 302)); L.append((cm, tm, sp[k], u))
        x = simlab.plant_step(x, u, caf[k], tf[k], simlab.TEXTBOOK)
    L = np.array(L); S.append(stats(L[:, 0], L[:, 1], L[:, 2], L[:, 3]))
print("columns: dTc std | rms(Ca_meas - sp) | innov Ca robust-sigma | innov T robust-sigma | innov Ca std | innov T std")
print("plant:", np.mean(P, 0).round(5)); print("sim  :", np.mean(S, 0).round(5))
