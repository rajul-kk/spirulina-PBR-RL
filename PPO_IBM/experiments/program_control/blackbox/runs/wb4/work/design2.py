from model import *
import math, sys, itertools
def run(X0, mu_max=0.04, Xt=560, late=(0.85, 0.45), n_end=2, Imax=1700, rpm_lo=70, rpm_mid=80, rpm_hi=80, x_hi=800,
        I0=500, Islope=5.0, ramp=120, tau=2.5):
    X, c, DO, T, Iacc, I = X0, 1.0, 8.0, 35.0, 200.0, 400.0
    tot = 0.0; S = 0.0; m_ = 0
    for k in range(7200):
        rpm = rpm_lo if X < 200 else (rpm_mid if X < x_hi else rpm_hi)
        tgt = min(Imax, I0 + Islope * X)
        I = min(tgt, I + ramp * 0.02) if tgt > I else tgt
        heat = I * 0.001 - 0.1 * (T - 25) + (rpm / 200) ** 3 * 0.1
        T += (heat + min(max(2 * (35 - T), -0.6), 2.0)) * 0.02
        mm = mu(I, X, rpm, c, T, DO, mu_max)
        fI, tm = f_light(I, X, rpm, c); d = max(tm - Iacc, 0); mm *= math.exp(-3e-6 * d * d)
        Iacc += 0.02 / tau * (tm - Iacc)
        X *= math.exp((mm - 0.01 * mu_max - 5e-4) * 0.02)
        DO = min(60, max(0, DO + 1.5 * X * mm * 0.02 - kla(rpm, X) * (DO - 8) * 0.02))
        stick = X / 300 * 0.05 * max(0.1, 1 - rpm / 250); brk = 0.5 * max(0, (rpm - 80) / 120) ** 2 * c ** 0.5 + 0.005 * (c - 1) ** 0.5
        c = max(1, c + (stick - brk - mm * (c - 1)) * 0.02)
        if k > 0 and k % 600 == 0:
            ev = k // 600; first_end = 12 - n_end
            if ev >= first_end: F = 0.5
            else:
                xt = Xt; j = first_end - ev
                if j <= len(late): xt *= late[len(late) - j]
                F = min(0.5, max(0, 1 - xt / X))
            tot += F * X * 20; X *= 1 - F
    return tot
if __name__ == "__main__":
    base = dict()
    arms = {
        "base": {},
        "hi100": dict(rpm_hi=100, x_hi=500), "hi120": dict(rpm_hi=120, x_hi=500), "hi140": dict(rpm_hi=140, x_hi=500),
        "mid90": dict(rpm_mid=90, rpm_hi=90), "lo55": dict(rpm_lo=55),
        "I1500": dict(Imax=1500), "I2000": dict(Imax=2000), "Xt420": dict(Xt=420), "Xt700": dict(Xt=700), "Xt850": dict(Xt=850),
        "late1": dict(late=(1.0, 0.6)), "late0": dict(late=(0.7, 0.35)), "nend3": dict(n_end=3, late=(0.85,)),
        "ramp250": dict(ramp=250), "Islope8": dict(Islope=8, I0=600),
    }
    for name, kw in arms.items():
        r = [run(X0, mu, **kw) for X0 in (19, 125, 250, 1500) for mu in (0.028, 0.04, 0.05)]
        print(name, " ".join("%5.1f" % (v / 1000) for v in r), " mean %.2f" % (sum(r) / len(r) / 1000), flush=True)
