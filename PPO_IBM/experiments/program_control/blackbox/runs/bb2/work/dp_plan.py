# Offline DP on a fitted growth model (from E01-E11 DW data) to find harvest policy structure.
# dX/dt = mu0*X*(1-X/2600)/(1+X/900); harvest at 12,24,...,132 h of fraction f<=0.5; V=20 L.
import numpy as np
def grow(X, mu0, h=12.0, n=60):
    dt = h / n
    for _ in range(n):
        X = X + dt * mu0 * X * (1 - X / 2600) / (1 + X / 900)
    return X
grid = np.exp(np.linspace(np.log(3), np.log(2600), 400))
fs = np.linspace(0, 0.5, 26)
def solve(mu0):
    G = grow(grid, mu0)
    V = np.zeros_like(grid)  # value after last harvest (132 h) = 0
    pol = []
    for k in range(11, 0, -1):  # harvest k at 12k h, then grow 12 h to next (except after last)
        best = np.full_like(grid, -1.0); bf = np.zeros_like(grid)
        for f in fs:
            rem = grid * (1 - f)
            if k < 11:
                nxt = grow(rem, mu0)
                v = 20 * f * grid + np.interp(nxt, grid, V)
            else:
                v = 20 * f * grid
            m = v > best; best[m] = v[m]; bf[m] = f
        V = best; pol.append((k, bf.copy()))
    return V, pol[::-1]
for mu0 in [0.018, 0.025, 0.032]:
    V, pol = solve(mu0)
    print('mu0', mu0)
    for k, bf in pol:
        # threshold density above which harvest starts, and hold level
        idx = np.where(bf > 0)[0]
        thr = grid[idx[0]] if len(idx) else None
        post = grid * (1 - bf)
        sel = [300, 600, 1000, 1500]
        print(f'  h{12*k:3d} harvest-start X={thr if thr is None else round(thr)}  f at X=' + ' '.join(f'{x}:{bf[np.argmin(abs(grid-x))]:.2f}' for x in sel))
    for X0 in [25, 75, 150, 300, 1000, 3000]:
        X12 = grow(min(X0 * 0.75, 2500), mu0)
        print('   inoc', X0, 'value', round(float(np.interp(X12, grid, V))))
