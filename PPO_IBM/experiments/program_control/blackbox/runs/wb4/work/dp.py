"""DP over the 11 harvest fractions, using a productivity curve P(X) from the reduced model
(calibrated to pilot data by factor CAL). State: density just before an event."""
import numpy as np, math, sys
from steady import eq
CAL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.93
RPM = float(sys.argv[2]) if len(sys.argv) > 2 else 80
Xs = np.exp(np.linspace(math.log(10), math.log(4000), 60))
P = []
for X in Xs:
    best = max(eq(I, X, RPM)[0] for I in range(400, 1801, 200))
    P.append(best * CAL)
P = np.array(P)
def Pf(X): return float(np.interp(math.log(X), np.log(Xs), P))
def grow(X, h=12.0):
    for _ in range(int(h / 0.25)):
        X += (Pf(X) - 5.5e-4 * X) * 0.25
    return X
grid = np.exp(np.linspace(math.log(10), math.log(4000), 300))
G = np.array([grow(x) for x in grid])
fr = np.linspace(0, 0.5, 51)
V = np.zeros(len(grid))          # after event 11: remaining growth wasted
pol = []
for k in range(11, 0, -1):
    Vn = np.zeros(len(grid)); pk = np.zeros(len(grid))
    for i, X in enumerate(grid):
        post = X * (1 - fr)
        nxt = np.interp(np.log(post), np.log(grid), G)
        val = fr * X * 20 + np.interp(np.log(np.clip(nxt, grid[0], grid[-1])), np.log(grid), V)
        j = int(np.argmax(val)); Vn[i] = val[j]; pk[i] = fr[j]
    V = Vn; pol.insert(0, pk)
print("mu at X=30,60:", Pf(30)/30, Pf(60)/60, " P(225,430,600,900,1300):", [round(Pf(x),2) for x in [225,430,600,900,1300]])
for k in range(11):
    # post-harvest density implied by the policy for pre-harvest X in a range
    s = []
    for X in [100, 200, 300, 450, 600, 800, 1000, 1500, 2500]:
        i = np.searchsorted(grid, X); s.append("%d:%.2f" % (X, pol[k][i]))
    print("event", k + 1, " ".join(s))
for X0 in [19, 62, 125, 250, 1000, 2750]:
    X = grow(X0); tot = 0
    for k in range(11):
        i = min(np.searchsorted(grid, X), len(grid) - 1); f = pol[k][i]
        tot += f * X * 20; X = grow(X * (1 - f))
    print("X0", X0, "DP total %.0f mg" % tot)
