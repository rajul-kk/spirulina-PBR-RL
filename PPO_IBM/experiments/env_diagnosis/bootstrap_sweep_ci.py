"""Bootstrap confidence intervals for held-out sweep logs.

`statistical_validation.md` put 95% CIs on the PPO/TD-MPC2-era sweeps and flagged the same
treatment for the TD3 line as outstanding. This applies it: parse the per-episode lines any
`td3_held_out_sweep.py` log already prints, and bootstrap the two gated yield statistics
(median and p25 of harvested_mg) plus crash rate.

Scope note, carried over from statistical_validation.md: this is EVALUATION-time variance
(one trained policy, N held-out seeds). It says nothing about TRAINING-seed variance, which
needs multiple full runs per configuration and is a separate, much more expensive question.

Adversarial cold starts (init <= 80 cells) are excluded from the yield statistics, matching
how the sweep itself scores them (survival, not yield).

Usage:
  python experiments/env_diagnosis/bootstrap_sweep_ci.py LOG [LOG ...]
  python experiments/env_diagnosis/bootstrap_sweep_ci.py --resamples 10000 /tmp/v54_sweep_final.log
"""

import argparse
import os
import re
import sys

import numpy as np

EP = re.compile(
    r"\[\s*(\d+)\s*/\s*(\d+)\]\s+seed=\s*(\d+)\s+.*?init=\s*(\d+)\s+steps=\s*(\d+)\s+"
    r"crashed=(\w+)\s+harvested=\s*([\d.]+)mg\s+time_avg_od=([\d.]+)"
)
SECTION = re.compile(r"^#{3,}\s*(.+?)\s*#{3,}\s*$")
ADVERSARIAL_MAX = 80
GATE = {"harvest": 90.0, "p25": 50.0, "crash": 0.08}


def parse(path):
    """Return [(section_name, [episode dicts])]. Handles multi-section logs and the
    duplicate-line printing artifact seen in some older logs."""
    sections, cur, name = [], [], os.path.basename(path)
    seen = set()
    with open(path, errors="replace") as fh:
        for line in fh:
            m = SECTION.match(line.strip())
            if m:
                if cur:
                    sections.append((name, cur))
                name, cur, seen = m.group(1), [], set()
                continue
            m = EP.search(line)
            if not m:
                continue
            idx, total, seed, init, steps, crashed, harv, od = m.groups()
            key = (idx, seed, harv, od)
            if key in seen:           # exact duplicate print of the same episode
                continue
            seen.add(key)
            cur.append({"seed": int(seed), "init": int(init),
                        "crashed": crashed.lower().startswith("t"),
                        "harv": float(harv), "od": float(od)})
    if cur:
        sections.append((name, cur))
    return sections


def boot_ci(vals, stat, resamples, rng, alpha=0.05):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx = rng.integers(0, len(vals), size=(resamples, len(vals)))
    dist = stat(vals[idx], axis=1)
    lo, hi = np.percentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(vals)), (float(lo), float(hi))


def verdict(point, lo, hi, gate):
    if lo >= gate:
        return "PASS (CI fully above gate)"
    if hi < gate:
        return "FAIL (CI fully below gate)"
    return f"AMBIGUOUS (CI straddles gate {gate:g})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for path in args.logs:
        if not os.path.exists(path):
            print(f"\n!! missing: {path}")
            continue
        for name, eps in parse(path):
            if not eps:
                continue
            yielding = [e for e in eps if e["init"] > ADVERSARIAL_MAX]
            adv = [e for e in eps if e["init"] <= ADVERSARIAL_MAX]
            harv = [e["harv"] for e in yielding]
            crash_all = [1.0 if e["crashed"] else 0.0 for e in eps]

            print("\n" + "=" * 88)
            print(f"  {os.path.basename(path)}  ::  {name}")
            print(f"  n={len(eps)} episodes ({len(yielding)} yield-scored, {len(adv)} adversarial)"
                  f"  |  {args.resamples:,} resamples, 95% percentile CI")
            print("=" * 88)
            if len(yielding) < 5:
                print("  too few yield-scored episodes for a meaningful CI")
                continue

            med, (mlo, mhi) = boot_ci(harv, np.median, args.resamples, rng)
            print(f"  harvested_mg median : {med:7.1f}  [{mlo:6.1f}, {mhi:6.1f}]   "
                  f"{verdict(med, mlo, mhi, GATE['harvest'])}")

            p25f = lambda a, axis=None: np.percentile(a, 25, axis=axis)
            p25, (plo, phi) = boot_ci(harv, p25f, args.resamples, rng)
            print(f"  harvested_mg p25    : {p25:7.1f}  [{plo:6.1f}, {phi:6.1f}]   "
                  f"{verdict(p25, plo, phi, GATE['p25'])}")

            cr, (clo, chi) = boot_ci(crash_all, np.mean, args.resamples, rng)
            print(f"  crash_rate          : {100*cr:6.1f}%  [{100*clo:5.1f}%, {100*chi:5.1f}%]"
                  f"   gate <= {100*GATE['crash']:.0f}%")

            od = [e["od"] for e in yielding]
            omed, (olo, ohi) = boot_ci(od, np.median, args.resamples, rng)
            print(f"  time_avg_od median  : {omed:7.4f}  [{olo:6.4f}, {ohi:6.4f}]  gate >= 0.011")

            print(f"  spread: min={min(harv):.1f}  max={max(harv):.1f}  "
                  f"IQR={p25f(harv):.1f}-{np.percentile(harv,75):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
