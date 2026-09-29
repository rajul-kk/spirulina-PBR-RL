"""Per-batch fit of (mu_eff, T_opt, tau_acclim) of my mean-field model to privileged true OD."""
import csv, json, math, sys
import numpy as np
from multiprocessing import Pool
from mf import f_light, ftemp, repair

res = {json.loads(l)["batch"]: json.loads(l) for l in open("../trials/results.jsonl")}

def sim(rows, harv, mueff, topt, tau, A0=200.0):
    od = float(rows[0]["true_od"]); c = 1.0; out = []; mem = 1.0; do2 = 8.0; A = A0
    for i in range(1, len(rows)):
        r = rows[i - 1]
        L, rpm, T = float(r["light_umol"]), float(r["stir_rpm"]), float(r["true_temp_c"])
        L2 = float(rows[i]["light_umol"])
        for j in range(50):
            Lj = L + (L2 - L) * j / 50
            X = od * 300
            fI, tm = f_light(Lj, X, rpm, c)
            A += 0.02 / tau * (tm - A)
            shock = math.exp(-3e-6 * max(tm - A, 0) ** 2)
            mix = min(max(rpm / 200, 0.25), 1); kla = (0.6 + 5 * mix ** 1.3) / (1 + (od / 10) ** 2)
            mu = mueff * fI * ftemp(T, topt) * repair(rpm) * shock / (1 + (do2 / 35) ** 4)
            net = mu - 0.0004 - 5e-4
            do2 += (1.5 * mu * X - kla * (do2 - 8)) * 0.02
            stick = od * 0.05 * max(0.1, 1 - rpm / 250); shear = max(0, (rpm - 80) / 120) ** 2
            brk = 0.5 * shear * math.sqrt(c) + 0.005 * math.sqrt(max(c - 1, 0))
            c = max(1, c + (stick - brk - (c - 1) * max(net, 0)) * 0.02)
            od *= math.exp(net * 0.02)
        h = float(rows[i]["hour"])
        if h in harv:
            f = harv[h] / (harv[h] + 20 * 300 * float(rows[i]["true_od"]))
            od *= 1 - f
        out.append(od)
    return out

def one(a):
    b, m, topt, tau = a
    rows = list(csv.DictReader(open(f"../trials/{b}.csv")))
    harv = {e["hour"]: e["harvested_mg"] for e in res[b]["harvests"] if e["harvested_mg"] > 0}
    true = np.array([float(r["true_od"]) for r in rows[1:]])
    s = np.array(sim(rows, harv, m, topt, tau)); n = min(len(s), len(true))
    return b, m, topt, tau, float(np.sqrt(np.mean(np.log(s[:n] / true[:n]) ** 2)))

if __name__ == "__main__":
    bs = sys.argv[1:]
    jobs = [(b, m, to, ta) for b in bs for m in np.arange(0.020, 0.048, 0.0015) for to in (34, 35, 36, 37, 38) for ta in (1.0, 2.5, 4.0)]
    with Pool(4) as p:
        R = p.map(one, jobs, chunksize=8)
    for b in bs:
        rb = sorted([r for r in R if r[0] == b], key=lambda r: r[4])
        best = rb[0]
        # best for topt=36,tau=2.5 for comparison
        ref = min([r for r in rb if r[2] == 36 and r[3] == 2.5], key=lambda r: r[4])
        print(f"{b}: mu {best[1]:.4f} Topt {best[2]} tau {best[3]} rms {best[4]:.4f} | ref(36,2.5) mu {ref[1]:.4f} rms {ref[4]:.4f}", flush=True)
