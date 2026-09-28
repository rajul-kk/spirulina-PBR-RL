# DP on my grey-box model: dX/dt = Pm*tanh(mu X/Pm) - m X ; harvests k=1..11 at 12k h, f in [0,0.5]; reward f*X*V
import numpy as np, sys
V = 20.0
def flow(X, mu, Pm, m, hrs=12.0, h=0.1):
    for _ in range(int(hrs / h)):
        X = X + h * (Pm * np.tanh(mu * X / Pm) - m * X)
    return X
def solve(mu=0.022, Pm=4.5, m=0.001, grid=np.exp(np.linspace(np.log(5), np.log(5000), 400)), fs=np.linspace(0, 0.5, 51), floor=40.0):
    Vn = np.zeros_like(grid)                       # value after harvest 11
    pol = np.zeros((12, len(grid)))
    G = flow(grid, mu, Pm, m)                      # post-harvest X -> next pre-harvest X
    Vnext_pre = None
    # value-to-go at pre-harvest state of harvest k: W_k(X) = max_f f X V + U_k((1-f)X), U_k(x) = W_{k+1}(G(x)), U_11 = 0
    U = lambda x: np.zeros_like(x)
    Ws = {}
    for k in range(11, 0, -1):
        Xpost = grid[:, None] * (1 - fs[None, :])
        val = fs[None, :] * grid[:, None] * V + U(Xpost)
        val[Xpost < floor] -= 1e9
        val[:, 0] = np.maximum(val[:, 0], U(grid * 1.0))    # f=0 always allowed
        j = val.argmax(1); pol[k] = fs[j]; W = val.max(1); Ws[k] = W.copy()
        Wk = W.copy()
        U = (lambda Wk: (lambda x: np.interp(np.log(np.maximum(flow(x, mu, Pm, m), 1e-3)), np.log(grid), Wk)))(Wk)
    return grid, pol, U
def simulate(X0, polfun, mu, Pm, m):
    X = flow(X0, mu, Pm, m); tot = 0
    for k in range(1, 12):
        f = polfun(k, X); tot += f * X * V; X = flow(X * (1 - f), mu, Pm, m)
    return tot / 1000
if __name__ == '__main__':
    grid, pol, _ = solve()
    for k in range(1, 12):
        # print Xpost threshold: smallest X where f>0 and resulting Xpost
        idx = np.where(pol[k] > 0)[0]
        s = ' '.join(f"{grid[i]:.0f}:{pol[k][i]:.2f}" for i in range(0, 400, 25))
        print(k, s)
    heur = lambda k, X, Xp=300, nd=4: 0.5 if k > 11 - nd else max(0, min(0.5, 1 - Xp / X))
    for (mu, Pm, m) in [(0.022, 4.5, 0.001), (0.015, 3.0, 0.001), (0.03, 6, 0.001), (0.022, 4.5, 0.0)]:
        g, P2, _ = solve(mu, Pm, m)
        dpf = lambda k, X, P2=P2, g=g: np.interp(np.log(X), np.log(g), P2[k])
        g0, P0, _ = solve()
        dpn = lambda k, X: np.interp(np.log(X), np.log(g0), P0[k])
        print(mu, Pm, m, ' '.join(f"X0={X0}: opt {simulate(X0, dpf, mu, Pm, m):.1f} nomDP {simulate(X0, dpn, mu, Pm, m):.1f} heur {simulate(X0, heur, mu, Pm, m):.1f} |" for X0 in [20, 140, 300, 3000]))
