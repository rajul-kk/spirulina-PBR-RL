"""Fit the textbook 2-state CSTR model to open-loop excitation batches.
dCa/dt = a(Caf-Ca) - k(T)Ca ; dT/dt = a(Tf-T) + beta k(T) Ca + alpha (Tc-T)
k(T) = exp(lnk - ER(1/T - 1/323)).  Time in minutes.
Disturbances Caf, Tf: piecewise constant on blocks of BL samples (free), TV-regularised.
Reads only ../trials/."""
import sys, glob, numpy as np
from scipy.optimize import least_squares
DT = 26.0/120
BL = int(sys.argv[2]) if len(sys.argv) > 2 else 8
pat = sys.argv[1] if len(sys.argv) > 1 else "excite"
files = sorted(glob.glob(f"../trials/b*_{pat}.csv"))
D = [np.genfromtxt(f, delimiter=",", names=True) for f in files]
Ca = np.array([d["Ca"] for d in D]); T = np.array([d["T"] for d in D]); Tc = np.array([d["Tc"] for d in D])
NB, N = Ca.shape; NBL = int(np.ceil(N/BL))
blk = np.arange(N)//BL

def rhs(ca, t, tc, caf, tf, p):
    a, lnk, ER, beta, alpha = p
    t = np.clip(t, 250.0, 400.0); ca = np.clip(ca, 0.0, 3.0)
    k = np.exp(lnk - ER*(1/t - 1/323.0))
    return a*(caf-ca) - k*ca, a*(tf-t) + beta*k*ca + alpha*(tc-t)

def sim(p, ca0, t0, caf, tf, lag=0, nsub=2):
    ca, t = ca0.copy(), t0.copy(); out_c = [ca]; out_t = [t]; h = DT/nsub
    for k in range(N-1):
        tc = Tc[:, max(k-lag, 0)]; cf = caf[:, blk[k]]; f = tf[:, blk[k]]
        for _ in range(nsub):
            k1 = rhs(ca, t, tc, cf, f, p); k2 = rhs(ca+h/2*k1[0], t+h/2*k1[1], tc, cf, f, p)
            k3 = rhs(ca+h/2*k2[0], t+h/2*k2[1], tc, cf, f, p); k4 = rhs(ca+h*k3[0], t+h*k3[1], tc, cf, f, p)
            ca = ca + h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); t = t + h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        out_c.append(ca); out_t.append(t)
    return np.array(out_c).T, np.array(out_t).T

def unpack(x):
    p = x[:5]; i = 5
    ca0 = x[i:i+NB]; i += NB; t0 = x[i:i+NB]; i += NB
    caf = x[i:i+NB*NBL].reshape(NB, NBL); i += NB*NBL
    tf = x[i:i+NB*NBL].reshape(NB, NBL)
    return p, ca0, t0, caf, tf

SC, ST = 0.003, 0.2
def res(x, lag=0, wtv=0.3):
    p, ca0, t0, caf, tf = unpack(x)
    c, t = sim(p, ca0, t0, caf, tf, lag)
    return np.concatenate([((c-Ca)/SC).ravel(), ((t-T)/ST).ravel(),
                           wtv*(np.diff(caf, axis=1)/0.01).ravel(), wtv*(np.diff(tf, axis=1)/1.0).ravel()])

p0 = np.array([1.0, np.log(7.2e10) - 8750/323.0, 8750.0, 209.2, 2.092])
x0 = np.concatenate([p0, Ca[:, 0], T[:, 0], np.ones(NB*NBL), 350*np.ones(NB*NBL)])
xs = np.concatenate([[1, 1, 1000, 100, 1], np.full(NB, .01), np.full(NB, 1.), np.full(NB*NBL, .01), np.full(NB*NBL, 1.)])
import time
from scipy.sparse import lil_matrix
nres = 2*NB*N + 2*NB*(NBL-1); S = lil_matrix((nres, len(x0)), dtype=int)
S[:, :5] = 1
for b in range(NB):
    rows = np.r_[b*N:(b+1)*N, NB*N+b*N:NB*N+(b+1)*N]
    cols = np.r_[5+b, 5+NB+b, 5+2*NB+b*NBL:5+2*NB+(b+1)*NBL, 5+2*NB+NB*NBL+b*NBL:5+2*NB+NB*NBL+(b+1)*NBL]
    for c in cols: S[rows, c] = 1
    o = 2*NB*N
    S[o+b*(NBL-1):o+(b+1)*(NBL-1), 5+2*NB+b*NBL:5+2*NB+(b+1)*NBL] = 1
    o2 = o + NB*(NBL-1)
    S[o2+b*(NBL-1):o2+(b+1)*(NBL-1), 5+2*NB+NB*NBL+b*NBL:5+2*NB+NB*NBL+(b+1)*NBL] = 1
t0_ = time.time(); res(x0); print("one residual eval: %.3fs" % (time.time()-t0_), flush=True)
for lag in (0, 1):
    r0 = res(x0, lag); n = 2*NB*N
    print(flush=True); print(f"lag {lag}: textbook-params rms (no dist fit): Ca {np.sqrt(np.mean(r0[:NB*N]**2))*SC:.4f}  T {np.sqrt(np.mean(r0[NB*N:n]**2))*ST:.3f}")
    s = least_squares(res, x0, x_scale=xs, args=(lag,), loss="soft_l1", max_nfev=40, jac_sparsity=S, verbose=1)
    p, ca0, t0, caf, tf = unpack(s.x); r = res(s.x, lag)
    print(f"lag {lag}: fitted rms Ca {np.sqrt(np.mean(r[:NB*N]**2))*SC:.4f}  T {np.sqrt(np.mean(r[NB*N:n]**2))*ST:.3f}  cost {s.cost:.1f}")
    print("   a=%.3f k323=%.4f (k0=%.3g) E/R=%.0f beta=%.1f alpha=%.3f" % (p[0], np.exp(p[1]), np.exp(p[1]+p[2]/323), p[2], p[3], p[4]))
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("   Caf blocks:\n", caf); print("   Tf blocks:\n", tf)
    np.save(f"fit_{pat}_lag{lag}.npy", s.x)
