"""Compare observed growth (true_od in privileged logs) with my model's predicted growth, replaying logged actions."""
import csv, sys, json, numpy as np
from sim import light_terms
from model import temp_factor
def load(b):
    rows = list(csv.DictReader(open(f"../trials/{b}.csv")))
    return [{k: float(v) for k, v in r.items()} for r in rows]
def analyse(b, mu=0.04, fconst=0.88*0.92*0.96, show=True):
    R = load(b)
    C = 1.0; out = []
    base = R[0]["turbidity_ntu"]/(250*R[0]["true_od"]/(1+0.05*R[0]["true_od"]))
    for i in range(len(R)-1):
        a, b2 = R[i], R[i+1]
        od = a["true_od"]; X = od*300
        # clump implied by turbidity (assuming pigment 1, drift=base)
        ratio = a["turbidity_ntu"]/(250*od/(1+0.05*od))/base
        Cimp = max(ratio, 0.3)**-3
        fI, tm = light_terms(X, a["light_umol"], a["stir_rpm"], max(Cimp,1)**(-1/3), 100, 2500)
        fT = temp_factor(a["true_temp_c"])
        rep = 1-0.35/(1+np.exp(-0.12*(a["stir_rpm"]-100)))
        pred = mu*fconst*fI*fT*rep
        dt = b2["hour"]-a["hour"]
        obs = np.log(b2["true_od"]/od)/dt if b2["pump_L"] == a["pump_L"] else float("nan")
        out.append((a["hour"], od, Cimp, a["light_umol"], a["stir_rpm"], fI, pred, obs, tm))
    if show:
        for o in out[::6]:
            print("h%5.1f od%6.3f Cimp%5.2f L%6.0f rpm%4.0f fI%5.2f pred%7.4f obs%7.4f tm%6.0f" % o)
    o = np.array(out); m = np.isfinite(o[:,7])
    print(b, "mean obs/pred ratio:", np.nanmean(o[m,7])/np.mean(o[m,6]))
    return o
if __name__ == "__main__":
    for b in sys.argv[1:]:
        analyse(b)
