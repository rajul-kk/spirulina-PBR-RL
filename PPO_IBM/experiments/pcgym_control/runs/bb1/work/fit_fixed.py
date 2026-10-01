"""Model check: hold kinetic/thermal parameters at literature (textbook) values, fit only disturbances
and initial states; compare cost with free-parameter fit. Also report disturbance step statistics."""
import sys, numpy as np
sys.argv = [sys.argv[0], "excite", "5"]
src = open("fit_model.py").read().split("import time")[0]
exec(src)
from scipy.sparse import lil_matrix
def run(pfix, label):
    def r2(z): return res(np.concatenate([pfix, z]), 0)
    z0 = x0[5:]
    nres = len(r2(z0)); S = lil_matrix((nres, len(z0)), dtype=int)
    for b in range(NB):
        rows = np.r_[b*N:(b+1)*N, NB*N+b*N:NB*N+(b+1)*N]
        cols = np.r_[b, NB+b, 2*NB+b*NBL:2*NB+(b+1)*NBL, 2*NB+NB*NBL+b*NBL:2*NB+NB*NBL+(b+1)*NBL]
        for c in cols: S[rows, c] = 1
        o = 2*NB*N; S[o+b*(NBL-1):o+(b+1)*(NBL-1), 2*NB+b*NBL:2*NB+(b+1)*NBL] = 1
        o2 = o+NB*(NBL-1); S[o2+b*(NBL-1):o2+(b+1)*(NBL-1), 2*NB+NB*NBL+b*NBL:2*NB+NB*NBL+(b+1)*NBL] = 1
    s = least_squares(r2, z0, x_scale=xs[5:], loss="soft_l1", max_nfev=40, jac_sparsity=S)
    p, ca0, t0, caf, tf = unpack(np.concatenate([pfix, s.x])); r = r2(s.x); n = NB*N
    print(f"{label}: cost {s.cost:.1f} rms Ca {np.sqrt(np.mean(r[:n]**2))*SC:.4f} T {np.sqrt(np.mean(r[n:2*n]**2))*ST:.3f}", flush=True)
    return caf, tf, r
caf, tf, r = run(p0, "textbook")
pf = np.load("fit_excite_lag0.npy")[:5]
run(pf, "fitted  ")
np.set_printoptions(precision=3, suppress=True, linewidth=220)
print("Caf (textbook params), blocks of 5 samples:\n", caf); print("Tf:\n", np.round(tf, 1))
n = NB*N
print("resid autocorr lag1: Ca %.2f T %.2f" % (np.corrcoef(r[:n-1], r[1:n])[0, 1], np.corrcoef(r[n:2*n-1], r[n+1:2*n])[0, 1]))
