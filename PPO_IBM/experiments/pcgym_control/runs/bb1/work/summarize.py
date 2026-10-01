"""Summarise trial logs: setpoint schedule, cost from logged Ca vs reported. Reads only ../trials/."""
import sys, glob, json, numpy as np
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
rep = {}
for l in open("../trials/results.jsonl"):
    r = json.loads(l); rep[r["batch"]] = r
for f in sorted(glob.glob(f"../trials/b*_{pat}.csv")):
    d = np.genfromtxt(f, delimiter=",", names=True); name = f.replace("\\", "/").split("/")[-1][:-4]
    sp = d["Ca_sp"]; ch = np.nonzero(np.diff(sp))[0] + 1
    c = np.mean(((d["Ca"] - sp)/0.01)**2)
    seg = " ".join(f"[{0 if i==0 else ch[i-1]}:{v:.3f}]" for i, v in enumerate(sp[np.r_[0, ch]]))
    sat = np.mean((d["Tc"] <= 295.01) | (d["Tc"] >= 301.99))
    print(f"{name}: rep {rep[name]['cost']:7.3f} logged {c:7.3f} Tmax {rep[name]['t_max']:.1f} sat {sat:.2f} Ca0 {d['Ca'][0]:.3f} T0 {d['T'][0]:.1f} sp {seg}")
