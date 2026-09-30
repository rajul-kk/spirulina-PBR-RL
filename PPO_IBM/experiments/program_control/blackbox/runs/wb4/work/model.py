"""Mean-field reduced model of the reactor, written by hand from the design source
(genetic_env.py). Used only for design studies; it does not import the simulator."""
import math
import numpy as np

L = 0.02 / 0.30   # light path m
ZQ = (np.arange(8) + 0.5) / 8.0 * L

def temp_factor(T, topt=36.0, tmin=10.0, tmax=44.5):
    if T <= tmin or T >= tmax: return 0.0
    num = (T - tmax) * (T - tmin) ** 2
    den = (topt - tmin) * ((topt - tmin) * (T - topt) - (topt - tmax) * (topt + tmin - 2.0 * T))
    return min(max(num / den, 0.0), 1.0)

def f_light(I, X, rpm, clump=1.0, Ks=100.0, Ki=2500.0, acc=None):
    k_sc = rpm * 0.004
    kr, kb, kg = 0.5 + 0.2 * X + k_sc, 0.2 + 0.25 * X + k_sc, 0.05 + 0.06 * X + k_sc
    qr = I * 0.4 * np.exp(-kr * ZQ)
    qt = qr + I * 0.4 * np.exp(-kb * ZQ) + I * 0.2 * np.exp(-kg * ZQ)
    s = clump ** (-1 / 3)
    gr, gt = s * qr, s * qt
    f_raw = (gr / (Ks + gr + gt ** 2 / Ki)).mean()
    rm, tm = gr.mean(), gt.mean()
    f_int = rm / (Ks + rm + tm ** 2 / Ki)
    w = 0.5 + 0.5 * min(max((rpm - 50) / 150, 0), 1)
    Ipk = math.sqrt(Ks * Ki); fmax = 0.4 * Ipk / (2 * Ks + 0.4 * Ipk)
    return min(max((w * f_int + (1 - w) * f_raw) / fmax, 0), 1), tm

def steady_T(I, rpm):
    heat = I * 0.001 + (rpm / 200) ** 3 * 0.1
    if heat <= 0.1 * 10 + 0.6: return 35.0
    return 25 + (heat - 0.6) / 0.1

def mu(I, X, rpm, clump=1.0, T=None, DO=8.0, mu_max=0.04, topt=36.0):
    fI, tm = f_light(I, X, rpm, clump)
    if T is None: T = steady_T(I, rpm)
    rep = 1 - 0.35 / (1 + math.exp(-0.12 * (rpm - 100)))
    shear = min(max((rpm - 80) / 100, 0), 1)
    fat = 1 - 0.15 * (0.5 * shear)     # equilibrium membrane damage
    fO2 = 1 / (1 + (DO / 35) ** 4)
    fQ, fpH = 0.9, 0.96
    return mu_max * fI * temp_factor(T, topt) * rep * fat * fO2 * fQ * fpH

def kla(rpm, X):
    m = min(max(rpm / 200, 0.25), 1.0)
    return (0.6 + 5 * m ** 1.3) * 1.05 / (1 + (X / 3000) ** 2)

def batch(policy, X0, mu_max=0.04, topt=36.0, T0=35.0):
    """policy(t_h, X_meas, c) -> (rpm, I, f_target_for_interval). Returns harvested mg total."""
    X, c, DO, T = X0, 1.0, 8.0, T0
    dt = 0.02; tot = 0.0; fsum = 0.0; n = 0; I_acc = 200.0
    for k in range(7200):
        rpm, I, f = policy(k * dt, X, c)
        fsum += f; n += 1
        # temp
        heat = I * 0.001 - 0.1 * (T - 25) + (rpm / 200) ** 3 * 0.1
        u = min(max(2 * (35 - T), -0.6), 2.0)
        T += (heat + u) * dt
        m = mu(I, X, rpm, c, T, DO, mu_max, topt)
        fI, tm = f_light(I, X, rpm, c)
        diff = max(tm - I_acc, 0); m *= math.exp(-3e-6 * diff * diff)
        I_acc += dt / 2.5 * (tm - I_acc)
        net = m - 0.01 * mu_max - 5e-4
        g = X * m * dt
        X *= math.exp(net * dt)
        DO += 1.5 * g - kla(rpm, X) * (DO - 8.0) * dt
        DO = min(max(DO, 0), 60)
        stick = X / 300 * 0.05 * max(0.1, 1 - rpm / 250)
        brk = 0.5 * max(0, (rpm - 80) / 120) ** 2 * c ** 0.5 + 0.005 * (c - 1) ** 0.5
        c = max(1.0, c + (stick - brk - max(m, 0) * (c - 1)) * dt)
        if k > 0 and k % 600 == 0:
            fr = fsum / n; fsum = 0; n = 0
            tot += fr * X * 20.0; X *= (1 - fr)
    return tot, X
