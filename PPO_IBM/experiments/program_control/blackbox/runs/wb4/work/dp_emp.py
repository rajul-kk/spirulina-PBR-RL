"""DP over harvest fractions using the EMPIRICAL productivity curve from pilot data (prod.py)."""
import numpy as np, math, sys
Xe = np.array([25, 75, 125, 187, 262, 350, 450, 550, 650, 750, 900, 1150, 1650, 3000])
Pe = np.array([0.80, 1.81, 3.05, 4.52, 5.88, 7.01, 7.86, 8.31, 8.78, 9.25, 9.33, 9.14, 7.51, 2.40])
S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
def Pf(X): return S * float(np.interp(math.log(X), np.log(Xe), Pe)) if X > 25 else S * 0.032 * X
def grow(X, h=12.0):
    for _ in range(int(h / 0.25)): X += Pf(X) * 0.25
    return X
grid = np.exp(np.linspace(math.log(10), math.log(4000), 300)); G = np.array([grow(x) for x in grid])
fr = np.linspace(0, 0.5, 51); V = np.zeros(len(grid)); pol = []
for k in range(11, 0, -1):
    Vn = np.zeros(len(grid)); pk = np.zeros(len(grid))
    for i, X in enumerate(grid):
        nxt = np.interp(np.log(X * (1 - fr)), np.log(grid), G)
        val = fr * X * 20 + np.interp(np.log(np.clip(nxt, grid[0], grid[-1])), np.log(grid), V)
        j = int(np.argmax(val)); Vn[i] = val[j]; pk[i] = fr[j]
    V = Vn; pol.insert(0, pk)
for k in range(11):
    s = []
    for X in [200, 300, 450, 600, 800, 1000, 1200, 1500, 2500]:
        i = np.searchsorted(grid, X); s.append("%d:%.2f" % (X, pol[k][i]))
    print("event", k + 1, " ".join(s))
def sim(X0, rule):
    X = grow(X0); tot = 0
    for k in range(11):
        f = rule(k + 1, X); tot += f * X * 20; X = grow(X * (1 - f))
    return tot
def dp_rule(ev, X): return pol[ev - 1][min(np.searchsorted(grid, X), len(grid) - 1)]
def v2_rule(Xt, late=(0.85, 0.45), n_end=2):
    def r(ev, X):
        fe = 12 - n_end
        if ev >= fe: return 0.5
        xt = Xt; j = fe - ev
        if j <= len(late): xt *= late[len(late) - j]
        return min(0.5, max(0, 1 - xt / X))
    return r
for X0 in [19, 62, 125, 156, 250, 1000, 2750]:
    print("X0 %4d DP %.1f g | v2 Xt400 %.1f Xt560 %.1f Xt800 %.1f Xt1000 %.1f | Xt800 late(.8,.5) %.1f  late(.9,.6) %.1f" % (
        X0, sim(X0, dp_rule) / 1e3, sim(X0, v2_rule(400)) / 1e3, sim(X0, v2_rule(560)) / 1e3, sim(X0, v2_rule(800)) / 1e3,
        sim(X0, v2_rule(1000)) / 1e3, sim(X0, v2_rule(800, (0.8, 0.5))) / 1e3, sim(X0, v2_rule(800, (0.9, 0.6))) / 1e3))
