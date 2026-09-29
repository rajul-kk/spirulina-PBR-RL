"""My own mean-field approximation of the plant equations (read from genetic_env.py).
No simulator import. Used to explore operating points offline."""
import numpy as np

LP = 20e-3 / 0.30  # light path m


def f_light(I, X, rpm, clump=1.0, Ks=100.0, Ki=2500.0):
    ks = rpm * 0.004
    zq = (np.arange(8) + 0.5) / 8.0 * LP
    kr, kb, kg = 0.5 + 0.2 * X + ks, 0.2 + 0.25 * X + ks, 0.05 + 0.06 * X + ks
    qr = I * 0.4 * np.exp(-kr * zq)
    qt = qr + I * 0.4 * np.exp(-kb * zq) + I * 0.2 * np.exp(-kg * zq)
    sh = clump ** (-1 / 3)
    g, q = sh * qr, sh * qt
    fraw = g / (Ks + g + q ** 2 / Ki)
    fresp = fraw.mean()
    rm, tm = g.mean(), q.mean()
    fint = rm / (Ks + rm + tm ** 2 / Ki)
    w = 0.5 + 0.5 * min(max((rpm - 50) / 150, 0), 1)
    Ip = np.sqrt(Ks * Ki)
    fmax = 0.4 * Ip / (2 * Ks + 0.4 * Ip)
    return min(max((w * fint + (1 - w) * fresp) / fmax, 0), 1), tm


def repair(rpm):
    return 1 - 0.35 / (1 + np.exp(-0.12 * (rpm - 100)))


def temp_eq(I, rpm):
    # steady temperature under thermostat
    heat = I * 0.001 + (rpm / 200) ** 3 * 0.1
    # at 35: need u = 0.1*10 - heat; u>=-0.6
    if heat - 1.0 <= 0.6:
        return 35.0
    return 25 + (heat - 0.6) / 0.1


def ftemp(T, topt=36.0, tmin=10.0, tmax=44.5):
    if T <= tmin or T >= tmax:
        return 0.0
    num = (T - tmax) * (T - tmin) ** 2
    den = (topt - tmin) * ((topt - tmin) * (T - topt) - (topt - tmax) * (topt + tmin - 2 * T))
    return min(max(num / den, 0), 1)


def fO2(X, mu, rpm):
    od = X / 300
    mix = min(max(rpm / 200, 0.25), 1)
    kla = (0.6 + 5 * mix ** 1.3) / (1 + (od / 10) ** 2)
    do2 = 8 + 1.5 * mu * X / kla
    return 1 / (1 + (do2 / 35) ** 4), do2


if __name__ == "__main__":
    import sys
    for rpm in (50, 70, 80, 90, 110):
        print("rpm", rpm)
        for od in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            X = od * 300
            best = None
            for I in range(100, 2001, 50):
                fi, tm = f_light(I, X, rpm)
                T = temp_eq(I, rpm)
                mu = 0.04 * 0.78 * fi * ftemp(T) * repair(rpm)
                fo, do2 = fO2(X, mu, rpm)
                mu *= fo
                p = mu * X * 20 * 12  # mg per 12h
                if best is None or p > best[0]:
                    best = (p, I, mu, fi, T, do2, tm)
            p, I, mu, fi, T, do2, tm = best
            print(f"  od {od:4.2f} I* {I:5d} mu {mu:.4f} fI {fi:.2f} T {T:.1f} DO {do2:.1f} pathmean {tm:.0f} prod/12h {p:6.0f} mg")
