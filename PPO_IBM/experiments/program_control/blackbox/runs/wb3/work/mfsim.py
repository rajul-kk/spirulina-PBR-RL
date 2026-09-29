"""Mean-field time simulation of my reading of the plant equations, driving a Controller
with synthetic sensor obs. My own simplified model (no simulator import)."""
import math
import numpy as np
from mf import f_light, ftemp, repair

LUMP = 0.81  # f_Q*f_P*f_carbon*f_pH lumped; fitted to 8 privileged trials (mu_eff mean 0.0323)


def run(ctrl_cls, inoc=300, seed=0, lump=LUMP, strain=None, verbose=False, log_every=0):
    r = np.random.RandomState(seed)
    mumax = max(0.02, r.normal(0.04, 0.006)); Ks = max(50, r.normal(100, 10)); Ki = max(500, r.normal(2500, 250))
    topt = float(np.clip(r.normal(36, 1), 30, 40)); tau = r.uniform(1, 4)
    if strain:
        mumax, topt, tau = strain.get("mumax", mumax), strain.get("topt", topt), strain.get("tau", tau)
    dr = r.uniform(0.95, 1.05)
    dens = min(inoc / 15000, 1)
    X = inoc * (1.25e8 - dens * 0.45e8) * 1e-7 / 20.0  # mg/L
    n = float(inoc)
    T = r.uniform(32, 38); integ = 0.0
    A = r.uniform(100, 300); clump = 1.0; pig = 1.0; mem = 1.0; do2 = 7.0; rpm_s = 50.0
    ctrl = ctrl_cls()
    hsum = 0.0; hcnt = 0; pump = 0.0; total = 0.0; harvests = []
    lost = False; logs = []
    for t in range(7200):
        od = X / 300
        turb = 250 * od * (0.7 + 0.3 * pig) * clump ** (-1 / 3) / (1 + 0.05 * od)
        turb = min(max(turb * (1 + 0.03 * rpm_s / 200 * r.randn()) + 0.05 * r.randn(), 0), 1000)
        obs = {"turbidity_ntu": turb * dr * r.uniform(0.98, 1.02), "ph": 9.9, "pump_L": pump,
               "conductivity": 27800.0, "temp_c": T * r.uniform(0.98, 1.02), "lux": 0.0, "t": t}
        stir, light, frac = ctrl.act(obs)
        stir = min(max(stir, 50), 200); light = min(max(light, 0), 2000); frac = min(max(frac, 0), 0.5)
        obs["lux"] = light * 30
        hsum += frac; hcnt += 1
        rpm_s = 0.9 * rpm_s + 0.1 * stir
        rpm = rpm_s
        # temperature
        T += light * 0.001 * 0.02 - 0.1 * (T - 25) * 0.02 + (rpm / 200) ** 3 * 0.1 * 0.02
        err = 35 - T; u_raw = 2 * err + 0.5 * integ; u = min(max(u_raw, -0.6), 2.0)
        if u == u_raw: integ += err * 0.02
        T += u * 0.02
        # clumping
        stick = od * 0.05 * max(0.1, 1 - rpm / 250)
        shear = max(0, (rpm - 80) / 120) ** 2
        brk = 0.5 * shear * math.sqrt(clump) + 0.005 * math.sqrt(max(clump - 1, 0))
        # light
        fI, tm = f_light(light, X, rpm, clump, Ks, Ki)
        A += 0.02 / tau * (tm - A)
        shock = math.exp(-3e-6 * max(tm - A, 0) ** 2)
        mix = min(max(rpm / 200, 0.25), 1)
        kla = (0.6 + 5 * mix ** 1.3) / (1 + (od / 10) ** 2)
        fo2 = 1 / (1 + (do2 / 35) ** 4)
        mem += -min(max((rpm - 80) / 100, 0), 1) * 0.05 * 0.02 + (1 - mem) * 0.1 * 0.02
        fat = 1 - 0.15 * (1 - mem)
        mu = mumax * lump * fI * ftemp(T, topt) * shock * fo2 * repair(rpm) * fat
        resp = 0.01 * mumax * (2 if light <= 1 else 1)
        net = mu - resp
        stress = min(max((resp - mu) / resp, 0), 1)
        lys = 5e-4 + 2e-3 * stress ** 2
        X *= math.exp(net * 0.02) * (1 - lys * 0.02)
        n *= math.exp(max(net, 0) * 0.02) * (1 - lys * 0.02)
        clump = max(1.0, clump + (stick - brk - (clump - 1) * max(net, 0)) * 0.02)
        do2 += (1.5 * mu * X - kla * (do2 - 8)) * 0.02
        pig += (-0.01 if (tm > 1000) else 0.01) * 0.02; pig = min(max(pig, 0.2), 1)
        if t > 0 and t % 600 == 0:
            f = hsum / hcnt; hsum = 0; hcnt = 0
            h = f * X * 20; total += h; X *= 1 - f; n *= 1 - f; pump += f * 20
            do2 = do2 * (1 - f) + 7 * f
            harvests.append((X / (1 - f) / 300 if f < 1 else 0, f, h))
        if log_every and t % log_every == 0:
            logs.append((t * 0.02, od, turb, T, light, stir, clump, shock, fI, do2))
        if n < 10 or X * 20 < 1:
            lost = True; break
    return {"total": total, "lost": lost, "harvests": harvests, "logs": logs,
            "strain": (mumax, topt, tau)}
