"""Replay logged batches through the controller's EKF (no plant calls) to inspect estimates/innovations."""
import sys, glob, json, numpy as np
from controller import Controller
pat = sys.argv[1]
rep = {json.loads(l)["batch"]: json.loads(l) for l in open("../trials/results.jsonl")}
for f in sorted(glob.glob(f"../trials/b*_{pat}.csv")):
    d = np.genfromtxt(f, delimiter=",", names=True); name = f.replace("\\", "/").split("/")[-1][:-4]
    c = Controller(); est = []; inn = []
    for k in range(len(d)):
        if k == 0:
            c.x = np.array([d["Ca"][0], d["T"][0], c.p["caf0"], c.p["tf0"]]); c.P = np.diag([c.p["r_ca"]**2, c.p["r_t"]**2, c.p["p0_caf"]**2, c.p["p0_tf"]**2])
        else:
            c._predict(d["Tc"][k-1]); inn.append((d["Ca"][k]-c.x[0], d["T"][k]-c.x[1])); c._update(d["Ca"][k], d["T"][k])
        est.append(c.x.copy())
    est = np.array(est); inn = np.array(inn); sp = d["Ca_sp"]
    cf = np.mean(((est[:, 0]-sp)/0.01)**2); cl = np.mean(((d["Ca"]-sp)/0.01)**2)
    print(f"{name}: rep {rep[name]['cost']:.3f} logged {cl:.3f} filtered {cf:.3f} | innov std Ca {inn[:,0].std():.4f} T {inn[:,1].std():.3f} | d2Ca/sqrt6 {np.diff(d['Ca'],2).std()/6**.5:.4f} d2T/sqrt6 {np.diff(d['T'],2).std()/6**.5:.3f} | Caf {est[::20,2].round(3)} Tf {est[::20,3].round(1)}")
