"""DP on fitted model: optimal harvest fraction at each 12h harvest (hours 12..132) vs density."""
import numpy as np, sys
th = np.load('fit_theta.npy')
V = 20.9
def grow(x, h=12.0, n=60):
    a, K, m = th
    for _ in range(n): x = x + h / n * (a * x / (1 + x / K) - m * x)
    return x
grid = np.exp(np.linspace(np.log(5), np.log(4000), 400))
fs = np.linspace(0, 0.5, 26)
Vn = np.zeros_like(grid)  # value after harvest 132: nothing
pol = []
for k in range(11, 0, -1):  # harvest k at hour 12k, then grow 12h to next (if k<11)
    nxt = Vn
    Vk = np.zeros_like(grid); Pk = np.zeros_like(grid)
    for i, x in enumerate(grid):
        best = -1; bf = 0
        for f in fs:
            xr = x * (1 - f)
            cont = 0.0 if k == 11 else np.interp(grow(xr), grid, nxt)
            val = f * x * V + cont
            if val > best + 1e-9: best, bf = val, f
        Vk[i] = best; Pk[i] = bf
    pol.append((12 * k, Pk)); Vn = Vk
pol = pol[::-1]
for X in [50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000]:
    print(f'X {X:5d}: ' + ' '.join(f'{np.interp(X, grid, P):.2f}' for h, P in pol))
# forward sims
for X0 in [20, 65, 130, 260, 650, 1600, 3200]:
    x = grow(X0); tot = 0; fl = []
    for h, P in pol:
        f = np.interp(x, grid, P); tot += f * x * V; fl.append(f); x = grow(x * (1 - f))
    print(f'X0 {X0:5d} tot {tot:8.0f}  f ' + ' '.join(f'{f:.2f}' for f in fl))
