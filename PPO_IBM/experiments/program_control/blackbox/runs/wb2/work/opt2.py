import itertools
from opt1 import P2, simulate
for n0 in [40, 200, 1500]:
    res = []
    for sp, kf, Lmax in itertools.product([1.5, 2.0, 2.5, 3.0, 4.0], [2, 3, 4], [1500, 1800, 2100]):
        tot, _ = simulate(P2(od_sp=sp, k_final=kf, rpm=75, Lmax=Lmax), n0=n0, mu_max=0.036)
        res.append((tot, sp, kf, Lmax))
    res.sort(reverse=True)
    print(n0, [f"{r[0]:.0f} sp{r[1]} kf{r[2]} L{r[3]}" for r in res[:8]], flush=True)
    for Lmax in [1500, 1800, 2100]:
        print("  L", Lmax, [f"sp{r[1]}kf{r[2]}:{r[0]:.0f}" for r in res if r[3] == Lmax and r[2] == 3])
