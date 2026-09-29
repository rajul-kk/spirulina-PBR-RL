"""DP over harvest fractions using the mean-field growth rate mu(X) (optimal light per X)."""
import numpy as np
from mf import f_light, temp_eq, ftemp, repair, fO2

def mu_of(X, rpm=80, mumax=0.04, topt=36.0, Imax=2000):
    best = (0, 0)
    for I in range(100, Imax + 1, 50):
        fi, tm = f_light(I, X, rpm)
        T = temp_eq(I, rpm)
        mu = mumax * 0.78 * fi * ftemp(T, topt) * repair(rpm)
        mu *= fO2(X, mu, rpm)[0]
        if mu > best[0]:
            best = (mu, I)
    return best

if __name__ == "__main__":
    for od in (3, 5, 7, 9):
        m, I = mu_of(od * 300)
        print("od", od, "mu", round(m, 4), "I", I, "prod/12h", round(m * od * 300 * 240))
    grid = np.exp(np.linspace(np.log(5), np.log(4000), 300))  # X mg/L
    mus = np.array([mu_of(x)[0] for x in grid])
    np.save("mu_grid.npy", np.vstack([grid, mus]))
    def grow(X, hours=12.0, dt=0.02):
        for _ in range(int(hours / dt)):
            X = X * np.exp(np.interp(np.log(X), np.log(grid), mus) * dt)
        return X
    G = np.array([grow(x) for x in grid])  # X after 12h growth
    fr = np.linspace(0, 0.5, 26)
    # events at 12..132 h (11). V[k](X_before_event_k)
    V = np.zeros_like(grid)  # after last event: remaining lost; but growth 132->144 irrelevant
    pol = []
    for k in range(11, 0, -1):
        Vn = np.full_like(grid, -1.0); Pn = np.zeros_like(grid)
        for i, X in enumerate(grid):
            for f in fr:
                Xa = X * (1 - f)
                if k == 11:
                    cont = 0.0
                else:
                    Xg = np.interp(np.log(Xa), np.log(grid), np.log(G)); Xg = np.exp(Xg)
                    cont = np.interp(np.log(Xg), np.log(grid), V)
                val = f * X * 20 + cont
                if val > Vn[i]:
                    Vn[i], Pn[i] = val, f
        V = Vn; pol.insert(0, Pn)
    for od0 in (0.06, 0.2, 0.4, 0.8, 2, 9):
        X = od0 * 300; X = grow(X); tot = 0; s = []
        for k in range(11):
            f = np.interp(np.log(X), np.log(grid), pol[k])
            f = fr[np.argmin(abs(fr - f))]
            tot += f * X * 20; s.append(f"{X/300:.2f}:{f:.2f}")
            X = grow(X * (1 - f))
        print(f"inoc od {od0}: total {tot:.0f} mg  ", " ".join(s))
