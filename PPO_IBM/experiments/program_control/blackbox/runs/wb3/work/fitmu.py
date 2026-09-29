"""Fit one effective growth scale per batch (mu_eff = mu_max*lump) of my mean-field model to the
privileged true-OD trajectory, driven by logged light/stir/temperature and the observed harvests."""
import csv, json, math, sys
import numpy as np
from mf import f_light, ftemp, repair

res = {json.loads(l)["batch"]: json.loads(l) for l in open("../trials/results.jsonl")}


def sim(rows, harv, mueff, topt=36.0, fo2=True, ret=False):
    od = float(rows[0]["true_od"]); c = 1.0; out = []; mem = 1.0; do2 = 8.0
    for i in range(1, len(rows)):
        r = rows[i - 1]
        L, rpm, T = float(r["light_umol"]), float(r["stir_rpm"]), float(r["true_temp_c"])
        for _ in range(50):
            X = od * 300
            fI, tm = f_light(L, X, rpm, c)
            mix = min(max(rpm / 200, 0.25), 1); kla = (0.6 + 5 * mix ** 1.3) / (1 + (od / 10) ** 2)
            f_o = 1 / (1 + (do2 / 35) ** 4) if fo2 else 1.0
            mem += -min(max((rpm - 80) / 100, 0), 1) * 0.001 + (1 - mem) * 0.002
            mu = mueff * fI * ftemp(T, topt) * repair(rpm) * f_o * (1 - 0.15 * (1 - mem))
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


def fit(b, topt=36.0):
    rows = list(csv.DictReader(open(f"../trials/{b}.csv")))
    harv = {e["hour"]: e["harvested_mg"] for e in res[b]["harvests"] if e["harvested_mg"] > 0}
    true = np.array([float(r["true_od"]) for r in rows[1:]])
    best = None
    for m in np.arange(0.018, 0.05, 0.001):
        s = np.array(sim(rows, harv, m, topt))
        n = min(len(s), len(true))
        e = np.sqrt(np.mean(np.log(s[:n] / true[:n]) ** 2))
        if best is None or e < best[0]:
            best = (e, m, s)
    e, m, s = best
    # residual by phase
    n = min(len(s), len(true)); lr = np.log(s[:n] / true[:n])
    seg = " ".join(f"{lr[k:k+24].mean():+.3f}" for k in range(0, n, 24))
    return m, e, seg


if __name__ == "__main__":
    for b in sys.argv[1:]:
        m, e, seg = fit(b)
        print(f"{b}: mu_eff {m:.3f} rms {e:.3f} | log-resid per day {seg}")
