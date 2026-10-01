"""Where does the cost come from? Split each batch into start-up (first 12 samples), 12 samples after each
setpoint change, and the rest. Plant logs: uses measured Ca shifted to the plant's scoring convention
(Ca[k+1] vs sp[k]) minus the noise bias 0.0018^2.  Reads only ../trials/."""
import sys, glob, json, numpy as np
pat = sys.argv[1]
rep = {json.loads(l)["batch"]: json.loads(l) for l in open("../trials/results.jsonl")}
tot = []
for f in sorted(glob.glob(f"../trials/b*_{pat}.csv")):
    d = np.genfromtxt(f, delimiter=",", names=True); name = f.replace("\\", "/").split("/")[-1][:-4]
    sp = d["Ca_sp"]; N = len(sp)
    e2 = ((d["Ca"][1:] - sp[:-1])/0.01)**2 - 0.0324
    ch = np.nonzero(np.diff(sp))[0] + 1
    lab = np.zeros(N-1, int); lab[:12] = 1
    for c in ch: lab[c:c+12] = 2
    parts = [e2[lab == i].sum()/N for i in range(3)]
    sat = np.mean((d["Tc"] <= 295.01) | (d["Tc"] >= 301.99))
    satss = np.mean(((d["Tc"] <= 295.01) | (d["Tc"] >= 301.99))[:-1][lab == 0])
    print(f"{name}: rep {rep[name]['cost']:.3f} est {sum(parts):.3f} = steady {parts[0]:.3f} + startup {parts[1]:.3f} + spchange {parts[2]:.3f} | sat {sat:.2f} sat(steady) {satss:.2f} |dsp| {np.abs(np.diff(sp)[ch-1]).round(3)} mean Tc {d['Tc'].mean():.1f}")
    tot.append([rep[name]["cost"]] + parts)
tot = np.array(tot); print("MEAN rep %.3f steady %.3f startup %.3f spchange %.3f; median rep %.3f" % (*tot.mean(0), np.median(tot[:, 0])))
