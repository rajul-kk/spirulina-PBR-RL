# DP with the interval-regression growth model (E01-E17): mu(x) = m0 - 0.026x + 0.0057x^2 (x g/L, L=2000),
# for x>1 linear decline to 0 at 2.6 g/L. Strain offset m0 varies.
import numpy as np
def mu(X, m0):
    x = X / 1000.0
    q = m0 - 0.026 * x + 0.0057 * x * x
    q1 = m0 - 0.026 + 0.0057
    return np.where(x <= 1.0, q, q1 * np.clip((2.6 - x) / 1.6, 0, None))
def grow(X, m0, h=12.0, n=48):
    dt = h / n
    for _ in range(n): X = X + dt * mu(X, m0) * X
    return X
grid = np.exp(np.linspace(np.log(3), np.log(2600), 500)); fs = np.linspace(0, 0.5, 51)
def solve(m0):
    V = np.zeros_like(grid); pol = {}
    for k in range(11, 0, -1):
        best = np.full_like(grid, -1.0); bf = np.zeros_like(grid)
        for f in fs:
            rem = grid * (1 - f)
            v = 20 * f * grid + (np.interp(grow(rem, m0), grid, V) if k < 11 else 0)
            m = v > best + 1e-9; best[m] = v[m]; bf[m] = f
        V = best; pol[k] = bf
    return V, pol
if __name__ == '__main__':
    for m0 in [0.022, 0.028, 0.034]:
        V, pol = solve(m0)
        print('m0', m0)
        for k in range(1, 12):
            post = [grid[i] * (1 - pol[k][i]) for i in range(len(grid))]
            sel = [100, 300, 600, 1000, 1500]
            print(f'  h{12*k:3d} post-harvest X for X=' + ' '.join(f'{x}->{grid[np.argmin(abs(grid-x))]*(1-pol[k][np.argmin(abs(grid-x))]):.0f}' for x in sel))
