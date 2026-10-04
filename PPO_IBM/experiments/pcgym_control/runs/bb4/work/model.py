"""Fitted plant model (from fit2.py on batches 0-5) + simulator used for controller tuning."""
import math, numpy as np
DT = 26.0 / 120; TREF = 322.0
P_FIT = dict(a=0.9697, lnk=-2.1531, E=8604.2, B=203.33, c=2.0808)
def deriv(Ca, T, Tc, Caf, Tf, p):
    k = math.exp(p['lnk'] - p['E'] * (1.0 / T - 1.0 / TREF))
    return p['a'] * (Caf - Ca) - k * Ca, p['a'] * (Tf - T) + p['B'] * k * Ca + p['c'] * (Tc - T)
def step(Ca, T, Tc, Caf, Tf, p, n=4):
    h = DT / n
    for _ in range(n):
        a1, b1 = deriv(Ca, T, Tc, Caf, Tf, p)
        a2, b2 = deriv(Ca + h/2*a1, T + h/2*b1, Tc, Caf, Tf, p)
        a3, b3 = deriv(Ca + h/2*a2, T + h/2*b2, Tc, Caf, Tf, p)
        a4, b4 = deriv(Ca + h*a3, T + h*b3, Tc, Caf, Tf, p)
        Ca += h/6*(a1 + 2*a2 + 2*a3 + a4); T += h/6*(b1 + 2*b2 + 2*b3 + b4)
    return Ca, T
def steady(Tc, Caf=1.0, Tf=350.5, p=P_FIT, x0=(0.9, 322.0)):
    Ca, T = x0
    for _ in range(600):
        Ca, T = step(Ca, T, Tc, Caf, Tf, p)
        if T > 400: break
    return Ca, T
if __name__ == '__main__':
    for Tc in [295, 296, 297, 298, 299, 300, 300.5, 301, 301.5, 302]:
        for Caf, Tf in [(1.0, 350.5), (1.02, 352.5), (0.98, 348.5)]:
            Ca, T = steady(Tc, Caf, Tf)
            print('Tc %.1f Caf %.2f Tf %.1f -> Ca %.4f T %.2f' % (Tc, Caf, Tf, Ca, T), end=' | ')
        print()
