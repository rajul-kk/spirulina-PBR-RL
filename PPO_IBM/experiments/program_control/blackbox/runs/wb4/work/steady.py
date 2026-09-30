from model import *
def eq(I, X, rpm):
    c, DO = 1.0, 8.0
    for _ in range(300):
        m = mu(I, X, rpm, c, None, DO)
        DO = 8 + 1.5 * m * X / kla(rpm, X)
        stick = X / 300 * 0.05 * max(0.1, 1 - rpm / 250)
        # solve c: stick = brk + m*(c-1)
        lo, hi = 1.0, 50.0
        for _ in range(40):
            cm = 0.5 * (lo + hi)
            brk = 0.5 * max(0, (rpm - 80) / 120) ** 2 * cm ** 0.5 + 0.005 * (cm - 1) ** 0.5
            if stick - brk - m * (cm - 1) > 0: lo = cm
            else: hi = cm
        c = 0.5 * (c + cm)
    return m * X, c, DO
if __name__ == "__main__":
    for rpm in [60, 80, 95, 110, 130]:
        print("rpm", rpm)
        for X in [100, 225, 400, 600, 900, 1300, 2000]:
            out = []
            for I in [800, 1200, 1600, 2000]:
                p, c, DO = eq(I, X, rpm); out.append("%5.2f(c%.1f,DO%2.0f)" % (p, c, DO))
            print(" X%5d " % X + " ".join(out))
