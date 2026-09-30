"""Confirmatory comparison on the 'final' split (docs/reports/comparison_protocol.md).

Inputs: results/final/<arm>__<run>.json, written by `harness.py ... --split final --out`.
Every file holds the same 200 episodes, so arms are compared per episode (yield-scored
episodes only, init > 80 cells). Uncertainty is hierarchical: each bootstrap draw resamples
the runs (seeds) of every arm and, jointly, the episodes, so the CI covers both run-to-run and
episode-to-episode spread. p-values for the pre-registered family are Holm-corrected.

  python compare.py
"""
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 20000
# Pre-registered family (protocol section 4); arms missing from results/final are skipped.
FAMILY = [("whitebox", "blackbox"), ("blackbox", "cmaes"), ("whitebox", "cmaes"),
          ("td3_lru", "td3_lstm"), ("td3_lru", "cmaes"), ("td3_lstm", "cmaes")]


def load():
    arms, crash = defaultdict(dict), defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(HERE, "*__*.json"))):
        arm, run = os.path.basename(f)[:-5].split("__", 1)
        eps = json.load(open(f))["episodes"]
        arms[arm][run] = {(e["seed"], e["init_cells"]): e["harvested_mg"] for e in eps if e["init_cells"] > 80}
        crash[arm][run] = float(np.mean([e["crashed"] for e in eps]))
    keys = sorted(next(iter(next(iter(arms.values())).values())))
    mats = {a: np.array([[r[k] for k in keys] for r in runs.values()]) / 1000 for a, runs in arms.items()}
    return mats, crash, keys


def boot_diff(A, B, rng):
    """Hierarchical bootstrap of mean_episodes(mean_runs(A) - mean_runs(B))."""
    n = A.shape[1]
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        e = rng.integers(0, n, n)
        ra = rng.integers(0, len(A), len(A))
        rb = rng.integers(0, len(B), len(B))
        out[i] = A[ra][:, e].mean() - B[rb][:, e].mean()
    return out


def seed_permutation_p(a, b):
    """Exact two-sided permutation test on per-run medians (for arms with several seeds)."""
    pooled, obs = np.concatenate([a, b]), abs(a.mean() - b.mean())
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        hits += abs(pooled[m].mean() - pooled[~m].mean()) >= obs - 1e-12
        total += 1
    return hits / total


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M, crash, keys = load()
    print(f"{len(keys)} yield-scored episodes per run\n")
    print(f"{'arm':<14}{'runs':>5}{'median g':>10}{'p25 g':>8}{'crash':>7}   per-run medians")
    for a, X in sorted(M.items(), key=lambda kv: -np.median(kv[1].mean(0))):
        med = np.median(X, axis=1)
        print(f"{a:<14}{len(X):>5}{np.median(X.mean(0)):>10.2f}{np.percentile(X.mean(0), 25):>8.2f}"
              f"{np.mean(list(crash[a].values())):>7.0%}   {np.round(med, 2).tolist()}")

    rows = [(a, b) for a, b in FAMILY if a in M and b in M]
    res = []
    for a, b in rows:
        d = boot_diff(M[a], M[b], rng)
        est = (M[a].mean(0) - M[b].mean(0)).mean()
        p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        res.append((a, b, est, *np.percentile(d, [2.5, 97.5]), p))
    if res:
        adj = holm(np.array([r[-1] for r in res]))
        print("\nPre-registered comparisons (mean paired difference, hierarchical 95% CI, Holm-adjusted p)")
        for (a, b, est, lo, hi, p), pa in zip(res, adj):
            frac = np.mean(M[a].mean(0) > M[b].mean(0))
            ptxt = f"{pa:.4f}" if pa >= 1 / N_BOOT else f"<{1 / N_BOOT:.0e}"
            print(f"  {a} - {b}: {est:+.2f} g [{lo:+.2f}, {hi:+.2f}]  p_holm={ptxt}  "
                  f"{a} better in {frac:.0%} of episodes")
    for a, b in rows:
        if len(M[a]) > 1 and len(M[b]) > 1:
            p = seed_permutation_p(np.median(M[a], 1), np.median(M[b], 1))
            print(f"\n  seed-level exact permutation test {a} vs {b} "
                  f"({len(M[a])} v {len(M[b])} runs, per-run medians): p={p:.4f}")
