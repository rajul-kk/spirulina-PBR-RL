"""Replay the logged actions of a pilot batch through my mean-field model and compare with the
privileged true OD; infer clump size from turbidity/true OD."""
import csv, json, sys, numpy as np
from mfmodel import Model
TR = "../trials/"
def load(b):
    rows = list(csv.DictReader(open(TR + b + ".csv")))
    return [{k: float(v) for k, v in r.items()} for r in rows]
def replay(b, inoc, harvests=None, mumax=0.04, topt=36.0, k=0.786, show=True, every=12):
    rows = load(b)
    m = Model(cells=inoc, mumax=mumax, topt=topt, k_const=k, T0=rows[0]["true_temp_c"])
    err = []
    out = []
    for i, r in enumerate(rows):
        if i and i % every == 0 or i == len(rows)-1:
            ratio = r["turbidity_ntu"] / (250 * r["true_od"] / (1 + 0.05 * r["true_od"]))
            out.append((r["hour"], r["true_od"], m.od, ratio, m.c ** (-1/3), r["stir_rpm"], r["light_umol"]))
        if r["hour"] % 12 != 0: err.append(np.log(max(m.od,1e-3) / r["true_od"]))
        for s in range(50):
            tt = m.step_count
            if harvests is not None and tt > 0 and tt % 600 == 0:
                hm = harvests[tt // 600 - 1]
                post = rows[tt // 50]["true_od"] * 6000.0
                m.hsum, m.hn = hm / max(hm + post, 1e-9), 0
                m.step(r["stir_rpm"], r["light_umol"], 0.0); m.hsum = 0; m.hn = 0
                # step() already added 0 to the fresh accumulator; keep it empty
                m.hsum, m.hn = 0.0, 0
            else:
                m.step(r["stir_rpm"], r["light_umol"], 0.0 if harvests is not None else r["harvest_frac"])
    if show:
        print(b, "hour trueOD modelOD turb/ideal model_c^-1/3 rpm light")
        for o in out: print("  %6.1f %6.3f %6.3f %5.3f %5.3f %5.0f %6.0f" % o)
    return float(np.sqrt(np.mean(np.square(err))))
if __name__ == "__main__":
    res = {json.loads(l)["batch"]: json.loads(l) for l in open(TR + "results.jsonl")}
    for b in sys.argv[1:]:
        print(replay(b, res[b]["inoculum"], [e["harvested_mg"] for e in res[b]["harvests"]]))
