"""Pre-registered secondary endpoints for the TD3 core comparison (docs/reports/comparison_protocol.md
section 4): steps to reach D1/D2, number of seeds reaching D1/D2, and crash rate of the final
checkpoint. Secondary endpoints: p-values are unadjusted (no Holm correction); lru vs lstm is the
pre-registered pair, the others are exploratory.

Inputs: results/rl_v3/rl-v3-<core>-s<seed>/session<N>/train_summary.log (curriculum history) and
results/final/td3_<core>__s<seed>.json, results/final/secondary/td3_<core>_best__s<seed>.json.
A seed whose last session lacks "--- Training Complete" is listed as incomplete and excluded from
the tests. Steps to D1/D2 are censored at the budget for seeds that never reach the tier.

  python td3_secondary.py
"""
import glob
import itertools
import json
import os
import re
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "rl_v3")
BUDGET = 2_000_000
CORE_ORDER = ["lru", "lstm", "gru", "rtu"]
PRE = [("lru", "lstm")]
EXPLORATORY = [("rtu", "lru"), ("gru", "lstm"), ("gru", "lru")]


def num(s):
    return int(s.replace(",", ""))


def parse_run(run_dir):
    """Replay the sessions in order; a RESUME rewinds the step counter and drops later transitions."""
    sessions = sorted(glob.glob(os.path.join(run_dir, "session*", "train_summary.log")),
                      key=lambda p: int(re.search(r"session(\d+)", p).group(1)))
    step, events, abort, complete, last_chunk = 0, [], False, False, 0
    for path in sessions:
        complete = False
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.search(r"\[RESUME\] step=([\d,]+)", line)
            if m:
                step = num(m.group(1))
                events = [e for e in events if e[0] <= step]
                continue
            m = re.match(r"\[Chunk\].*\bsteps=([\d,]+)", line)
            if m:
                step += num(m.group(1))
                last_chunk = step
                continue
            m = re.search(r"Curriculum (ADVANCED|DEMOTED): D(\d) -> D(\d)", line)
            if m:
                events.append((step, m.group(1), int(m.group(2)), int(m.group(3))))
            if "[CAPABILITY ABORT]" in line:
                abort = True
            if "--- Training Complete" in line:
                complete = True
    first = {}
    for st, kind, _, to in events:
        if kind == "ADVANCED":
            first.setdefault(to, st)
    return dict(d1=first.get(1), d2=first.get(2), final_tier=events[-1][3] if events else 0,
                demotions=sum(e[1] == "DEMOTED" for e in events), abort=abort, steps=step,
                complete=complete)


def load_summary(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))["summary"]


def discover():
    runs = []
    for d in glob.glob(os.path.join(LOGS, "rl-v3-*-s*")):
        m = re.fullmatch(r"rl-v3-([a-z0-9]+)-s(\d+)", os.path.basename(d))
        if not m:
            continue
        core, seed = m.group(1), int(m.group(2))
        r = parse_run(d)
        fin = load_summary(os.path.join(HERE, f"td3_{core}__s{seed}.json"))
        best = load_summary(os.path.join(HERE, "secondary", f"td3_{core}_best__s{seed}.json"))
        r.update(core=core, seed=seed, crash=fin["crash_rate"] if fin else None,
                 final_med=fin["median_mg"] / 1000 if fin else None,
                 best_med=best["median_mg"] / 1000 if best else None)
        runs.append(r)
    rank = lambda c: CORE_ORDER.index(c) if c in CORE_ORDER else len(CORE_ORDER)
    return sorted(runs, key=lambda r: (rank(r["core"]), r["core"], r["seed"]))


def fisher_two_sided(k1, n1, k2, n2):
    K, N = k1 + k2, n1 + n2
    prob = lambda k: comb(n1, k) * comb(n2, K - k) / comb(N, K)
    obs = prob(k1)
    return sum(prob(k) for k in range(max(0, K - n2), min(n1, K) + 1) if prob(k) <= obs * (1 + 1e-9))


def perm_p(a, b):
    """Exact two-sided permutation test on the difference in means over all splits of the pooled seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, obs = np.concatenate([a, b]), abs(a.mean() - b.mean())
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        hits += abs(pooled[m].mean() - pooled[~m].mean()) >= obs - 1e-12
        total += 1
    return hits / total


def fmt_step(s):
    return f"{s:,}" if s is not None else "-"


def fmt_med(vals):
    v = np.median(vals)
    return f"{v:,.0f}" if v < BUDGET else f">={BUDGET:,}"


if __name__ == "__main__":
    out = []
    P = lambda s="": (print(s), out.append(s))

    runs = discover()
    P("TD3 secondary endpoints (protocol section 4). Unadjusted p-values, no Holm correction;")
    P(f"lru vs lstm is the pre-registered pair, other pairs are exploratory. Budget {BUDGET:,} steps.\n")
    P(f"{'core':<6}{'seed':>5}{'D1 at':>11}{'D2 at':>11}{'tier':>6}{'dem':>5}{'abort':>7}{'steps':>11}"
      f"{'crash%':>8}{'final g':>9}{'best g':>8}  status")
    for r in runs:
        f = lambda v, p: f"{v:.{p}f}" if v is not None else "-"
        P(f"{r['core']:<6}{r['seed']:>5}{fmt_step(r['d1']):>11}{fmt_step(r['d2']):>11}"
          f"{'D' + str(r['final_tier']):>6}{r['demotions']:>5}{'yes' if r['abort'] else 'no':>7}{r['steps']:>11,}"
          f"{f(r['crash'] * 100 if r['crash'] is not None else None, 1):>8}{f(r['final_med'], 2):>9}"
          f"{f(r['best_med'], 2):>8}  {'complete' if r['complete'] else 'INCOMPLETE (excluded)'}")

    by = {}
    for r in runs:
        if r["complete"]:
            by.setdefault(r["core"], []).append(r)
    P("\nPer core (complete seeds only)")
    P(f"{'core':<6}{'seeds':>6}{'reach D1':>10}{'reach D2':>10}{'med steps D1':>15}{'med steps D2':>15}{'mean crash':>12}")
    for c, rs in by.items():
        cs = lambda k: [min(r[k], BUDGET) if r[k] is not None else BUDGET for r in rs]
        crash = [r["crash"] for r in rs if r["crash"] is not None]
        P(f"{c:<6}{len(rs):>6}{sum(r['d1'] is not None for r in rs):>10}{sum(r['d2'] is not None for r in rs):>10}"
          f"{fmt_med(cs('d1')):>15}{fmt_med(cs('d2')):>15}"
          f"{(f'{np.mean(crash):.1%}' if crash else '-'):>12}")
    P("(medians use steps censored at the budget; '>=' means at least half the seeds never reached the tier)")

    pairs = [(a, b, "pre-registered") for a, b in PRE] + [(a, b, "exploratory") for a, b in EXPLORATORY]
    for a, b, tag in pairs:
        if a not in by or b not in by or len(by[a]) < 2 or len(by[b]) < 2:
            continue
        A, B = by[a], by[b]
        P(f"\n{a} vs {b} ({tag}; {len(A)} v {len(B)} seeds)")
        for tier, key in (("D1", "d1"), ("D2", "d2")):
            ka, kb = sum(r[key] is not None for r in A), sum(r[key] is not None for r in B)
            P(f"  seeds reaching {tier}: {a} {ka}/{len(A)}, {b} {kb}/{len(B)}  Fisher exact two-sided p={fisher_two_sided(ka, len(A), kb, len(B)):.4f}")
        for tier, key in (("D1", "d1"), ("D2", "d2")):
            ca = [min(r[key], BUDGET) if r[key] is not None else BUDGET for r in A]
            cb = [min(r[key], BUDGET) if r[key] is not None else BUDGET for r in B]
            P(f"  steps to {tier} (censored at budget): mean {a} {np.mean(ca):,.0f}, {b} {np.mean(cb):,.0f}; "
              f"median {a} {fmt_med(ca)}, {b} {fmt_med(cb)}; exact permutation p={perm_p(ca, cb):.4f}")
        xa = [r["crash"] for r in A if r["crash"] is not None]
        xb = [r["crash"] for r in B if r["crash"] is not None]
        if len(xa) == len(A) and len(xb) == len(B):
            P(f"  final-checkpoint crash rate: mean {a} {np.mean(xa):.1%}, {b} {np.mean(xb):.1%}; "
              f"seed-level exact permutation p={perm_p(xa, xb):.4f}")
        else:
            P("  final-checkpoint crash rate: missing result files, test skipped")

    open(os.path.join(HERE, "td3_secondary.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
