"""Offline stress test on my own model: plant differs from the controller's model."""
import numpy as np
import plantsim
C = plantsim.load("controller.py")
seeds = range(500000, 500030)
base = (plantsim.UA, plantsim.K0, plantsim.NOISE_CA, plantsim.NOISE_T)
for name, ua, k0, nca, nT in [("nominal", 1, 1, 1, 1), ("UA -10%", 0.9, 1, 1, 1), ("UA +10%", 1.1, 1, 1, 1),
                              ("k0 +10%", 1, 1.1, 1, 1), ("k0 -10%", 1, 0.9, 1, 1), ("noise x3", 1, 1, 3, 3)]:
    plantsim.UA, plantsim.K0, plantsim.NOISE_CA, plantsim.NOISE_T = base[0] * ua, base[1] * k0, base[2] * nca, base[3] * nT
    res = [plantsim.run(C(), s) for s in seeds]
    c = np.array([r[0] for r in res])
    print(f"{name:10s} mean={c.mean():.3f} median={np.median(c):.3f} max={c.max():.3f} runaways={sum(r[1] for r in res)} tmax={max(r[2] for r in res):.1f}", flush=True)
