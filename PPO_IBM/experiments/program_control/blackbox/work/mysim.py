# my own grey-box model identified from pilot data (NOT the plant): dX/dt = Pm*tanh(mu*X/Pm); harvest every 12h at 12..132
import numpy as np, itertools
V = 20.0
def grow(X, hrs, mu=0.025, Pm=5.0):
    for _ in range(int(hrs / 0.1)):
        X += 0.1 * Pm * np.tanh(mu * X / Pm)
    return X
def run(X0, fpol, mu=0.025, Pm=5.0):
    X = X0; tot = 0
    for k in range(1, 12):
        X = grow(X, 12, mu, Pm)
        f = fpol(k, X)
        tot += f * X * V; X *= (1 - f)
    return tot / 1000
def pol(Xp, ndump):
    def f(k, X):
        if k > 11 - ndump: return 0.5
        return float(np.clip(1 - Xp / X, 0, 0.5))
    return f
for X0 in [20, 70, 140, 300, 1000, 2500]:
    for (mu, Pm) in [(0.025, 5.0), (0.018, 3.0)]:
        best = max(((run(X0, pol(Xp, nd), mu, Pm), Xp, nd) for Xp in [100, 150, 200, 250, 300, 400, 600] for nd in [1, 2, 3, 4, 5]))
        print(f"X0 {X0:5d} mu {mu} Pm {Pm}: best {best[0]:.1f} g at Xpost {best[1]} ndump {best[2]} | Xp200/nd2 {run(X0,pol(200,2),mu,Pm):.1f} Xp300/nd3 {run(X0,pol(300,3),mu,Pm):.1f}")
