# Planning model: dOD/dt = G*tanh(mu*od/G), harvest events every 12 h (11 events, last at 132 h), batch 144 h.
import numpy as np, sys
def grow(od, hours, mu=0.03, G=0.034, dt=0.1):
    for _ in range(int(hours/dt)): od += G*np.tanh(mu*od/G)*dt
    return od
def dp(od0, mu=0.03, G=0.034):
    grid = np.exp(np.linspace(np.log(0.02), np.log(15), 400))
    nxt = np.array([grow(x, 12, mu, G) for x in grid])
    V = np.zeros_like(grid)   # value after last event: nothing
    hs = np.linspace(0, 0.5, 26); pol = []
    for k in range(11, 0, -1):   # event k at 12k h; state = od just before event
        newV = np.zeros_like(grid); best = np.zeros_like(grid)
        for i, x in enumerate(grid):
            vals = []
            for h in hs:
                rem = x*(1-h)
                if k == 11: fut = 0
                else: fut = np.interp(np.interp(rem, grid, nxt), grid, V)
                vals.append(x*h*6000 + fut)
            j = int(np.argmax(vals)); newV[i] = vals[j]; best[i] = hs[j]
        V = newV; pol.append((k, best))
    return grid, V, dict(pol), nxt
if __name__ == '__main__':
    mu, G = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03, float(sys.argv[2]) if len(sys.argv) > 2 else 0.034
    grid, V, pol, nxt = dp(0, mu, G)
    for od0 in (0.06, 0.2, 0.41, 0.8, 2.0, 5.0, 9.0):
        x = grow(od0, 12, mu, G); tot = 0; s = []
        for k in range(1, 12):
            h = np.interp(x, grid, pol[k]); tot += x*h*6000; s.append(f"{x:.2f}/{h:.2f}")
            x = grow(x*(1-h), 12, mu, G)
        print(f"od0 {od0}: total {tot:.0f} mg  ", ' '.join(s))
