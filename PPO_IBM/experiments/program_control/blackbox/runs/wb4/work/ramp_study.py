from design2 import run
import sys
arms = {"base r120": {}, "r60": dict(ramp=60), "r30": dict(ramp=30), "r250": dict(ramp=250),
        "I0=350 s5": dict(I0=350), "I0=600 s4": dict(I0=600, Islope=4)}
for name, kw in arms.items():
    r = [run(X0, 0.04, Xt=800, x_hi=1100, tau=tau, **kw) for X0 in (19, 62, 156) for tau in (1.0, 4.0)]
    print(name, " ".join("%5.2f" % (v / 1000) for v in r), " mean %.3f" % (sum(r) / len(r) / 1000), flush=True)
