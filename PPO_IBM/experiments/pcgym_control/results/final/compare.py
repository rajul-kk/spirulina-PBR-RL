"""Confirmatory comparison on the PC-Gym final split (docs/reports/pcgym_protocol.md).

Inputs: results/final/<arm>__<run>.json (harness.py --split final --out, or rl_sac.py's
final.json copied in). Cost is lower-is-better, so "A - B" below zero means A is better. The
statistics are the ones used for the photobioreactor comparison: hierarchical bootstrap over
runs and episodes, Holm-corrected, plus an exact seed-level permutation test.

  python compare.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "program_control", "results", "final"))
import compare as pc  # noqa: E402  (boot_diff, holm, seed_permutation_p)

FAMILY = [("whitebox", "blackbox"), ("blackbox", "cmaes-pid"), ("whitebox", "cmaes-pid"),
          ("blackbox", "sac"), ("whitebox", "sac")]

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    arms, runaway = defaultdict(dict), defaultdict(list)
    for f in sorted(glob.glob(os.path.join(HERE, "*__*.json"))):
        arm, run = os.path.basename(f)[:-5].split("__", 1)
        eps = json.load(open(f))["episodes"]
        arms[arm][run] = {e["seed"]: e["cost"] for e in eps}
        runaway[arm].append(np.mean([e["runaway"] for e in eps]))
    keys = sorted(next(iter(next(iter(arms.values())).values())))
    M = {a: np.array([[r[k] for k in keys] for r in runs.values()]) for a, runs in arms.items()}
    print(f"{len(keys)} episodes per run; cost = mean ((Ca - setpoint) / 0.01)^2, lower is better\n")
    print(f"{'arm':<12}{'runs':>5}{'mean cost':>11}{'median':>9}{'runaway':>9}   per-run mean cost")
    for a, X in sorted(M.items(), key=lambda kv: kv[1].mean()):
        print(f"{a:<12}{len(X):>5}{X.mean():>11.3f}{np.median(X.mean(0)):>9.3f}{np.mean(runaway[a]):>9.1%}   "
              f"{np.round(X.mean(1), 3).tolist()}")
    rows = [(a, b) for a, b in FAMILY if a in M and b in M]
    res = []
    for a, b in rows:
        d = pc.boot_diff(M[a], M[b], rng)
        p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        res.append((a, b, (M[a].mean(0) - M[b].mean(0)).mean(), *np.percentile(d, [2.5, 97.5]), p))
    if res:
        adj = pc.holm(np.array([r[-1] for r in res]))
        print("\nPre-registered comparisons (mean paired cost difference, hierarchical 95% CI, Holm-adjusted p)")
        for (a, b, est, lo, hi, p), pa in zip(res, adj):
            ptxt = f"{pa:.4f}" if pa >= 1 / pc.N_BOOT else f"<{1 / pc.N_BOOT:.0e}"
            print(f"  {a} - {b}: {est:+.3f} [{lo:+.3f}, {hi:+.3f}]  p_holm={ptxt}  "
                  f"{a} better in {np.mean(M[a].mean(0) < M[b].mean(0)):.0%} of episodes")
    for a, b in rows:
        if len(M[a]) > 1 and len(M[b]) > 1:
            print(f"\n  seed-level exact permutation test {a} vs {b} ({len(M[a])} v {len(M[b])} runs, "
                  f"per-run mean cost): p={pc.seed_permutation_p(M[a].mean(1), M[b].mean(1)):.4f}")
