"""Analysis of the training-context ablation (docs/reports/comparison_protocol.md section 9).

Pre-registered family, Holm-corrected over two (endpoint and statistics as section 4):
  C1  BI-LSTM   vs LSTM control
  C2  W150-LSTM vs LSTM control
Endpoint: mean paired per-episode harvest difference (g) on the `final` split, over yield-scored
episodes (initial culture > 80 cells), of each run's SELECTED checkpoint (selection.json, made on
`val` only). Hierarchical bootstrap 95% CI (compare.boot_diff, 20,000 draws, seed 0), Holm over
the two, and the seed-level exact permutation test on per-run medians (the headline, section 8).
Secondary, unadjusted: LRU control vs BI-LSTM and vs W150-LSTM; the same four comparisons on the
raw final checkpoints (continuity with section 4); crash rates; seeds reaching D1/D2 and steps
to D1/D2 (training logs, censored at 2M).

Arms: lstmctl, lructl (controls), bi, w150 (new).
Inputs, relative to results/final:
  controls  final ckpt: td3_lstm__s<n>.json / td3_lru__s<n>.json
            best ckpt : secondary/td3_lstm_best__s<n>.json / secondary/td3_lru_best__s<n>.json
  new arms  ctx/td3_<arm>_<final|best>__s<n>.final.json
  ctx/selection.json  (arm -> seed -> selected), from selection.py
  logs (under --logs-dir, default results): rl_v3/rl-v3-<lstm|lru>-s<n>, rl_ctx/<arm>-s<n>

Every input must exist. Otherwise the missing inputs are listed, nothing is computed and the exit
code is 1, unless --allow-partial (development only; the output is then labelled NOT the
pre-registered result and runs only the comparisons whose arms are complete).

  python compare_ctx.py                  # print only
  python compare_ctx.py --write          # also write ctx/compare_ctx.txt
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(HERE)
LOGS_DIR = os.path.dirname(FINAL_DIR)
SEEDS = [1, 2, 3, 4, 5]
BUDGET = 2_000_000
OUT_NAME = "compare_ctx.txt"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse, never copy: statistics from compare.py, log replay / Fisher / permutation from
# compare_followup.py and td3_secondary.py.
cf = _load_module("pc_compare_followup", os.path.join(FINAL_DIR, "followup", "compare_followup.py"))
cmp, sec = cf.cmp, cf.sec

ARMS = ["lstmctl", "lructl", "bi", "w150"]
LABEL = {"lstmctl": "LSTM-ctl", "lructl": "LRU-ctl", "bi": "BI-LSTM", "w150": "W150-LSTM"}
CONTROL_CORE = {"lstmctl": "lstm", "lructl": "lru"}
FAMILY = [("C1", "bi", "lstmctl"), ("C2", "w150", "lstmctl")]
SECONDARY = [("S1", "lructl", "bi"), ("S2", "lructl", "w150")]
CONTINUITY = [("C1", "bi", "lstmctl"), ("C2", "w150", "lstmctl"),
              ("S1", "lructl", "bi"), ("S2", "lructl", "w150")]


def ckpt_path(arm, ck, seed):
    """Final-split per-episode JSON of one checkpoint, relative to results/final."""
    if arm in CONTROL_CORE:
        core = CONTROL_CORE[arm]
        return (f"td3_{core}__s{seed}.json" if ck == "final"
                else f"secondary/td3_{core}_best__s{seed}.json")
    return f"ctx/td3_{arm}_{ck}__s{seed}.final.json"


def log_dirs(arm, seed):
    if arm in CONTROL_CORE:
        return [f"rl_v3/rl-v3-{CONTROL_CORE[arm]}-s{seed}"]
    return [f"rl_ctx/{arm}-s{seed}"]


def load_selection(final_dir):
    p = os.path.join(final_dir, "ctx", "selection.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def load_arm(arm, final_dir, selection):
    """-> dict(sel={seed: g-dict}, raw={seed: g-dict}, crash_sel, crash_raw, picks, missing)."""
    a = dict(sel={}, raw={}, crash_sel={}, crash_raw={}, picks={}, missing=[])
    for s in SEEDS:
        rel = ckpt_path(arm, "final", s)
        if os.path.exists(os.path.join(final_dir, rel)):
            a["raw"][s], a["crash_raw"][s] = cf.read_result(os.path.join(final_dir, rel))
        else:
            a["missing"].append(rel)
        pick = selection.get(arm, {}).get(str(s), {}).get("selected")
        if pick is None:
            a["missing"].append(f"ctx/selection.json entry {arm}/{s}")
            continue
        a["picks"][s] = pick
        rel = ckpt_path(arm, pick, s)
        if pick == "final" and s in a["raw"]:
            a["sel"][s], a["crash_sel"][s] = a["raw"][s], a["crash_raw"][s]
        elif os.path.exists(os.path.join(final_dir, rel)):
            a["sel"][s], a["crash_sel"][s] = cf.read_result(os.path.join(final_dir, rel))
        elif rel not in a["missing"]:
            a["missing"].append(rel)
    return a


def matrices(arms, field):
    """Per-arm (runs x episodes) matrices in g over the episodes common to every loaded run."""
    key_sets = [set(r) for a in arms.values() for r in a[field].values()]
    notes = []
    if not key_sets:
        return {}, [], notes
    keys = sorted(set.intersection(*key_sets))
    if any(len(k) != len(keys) for k in key_sets):
        notes.append(f"WARNING: runs do not share one episode set; using the {len(keys)} common episodes")
    mats = {n: np.array([[a[field][s][k] for k in keys] for s in sorted(a[field])])
            for n, a in arms.items() if a[field]}
    return mats, keys, notes


def parse_run(logs_dir, arm, seed):
    paths = [p for d in log_dirs(arm, seed) for p in cf.session_files(os.path.join(logs_dir, d))]
    if not paths:
        return None
    r = cf.replay_logs(paths)
    for k in ("d1", "d2"):
        if r[k] is not None and r[k] > BUDGET:
            r[k] = None
    r["finished"] = r["complete"] and (r["steps"] >= BUDGET or r["abort"])
    return r


def boot_compare(mats, a, b, rng):
    d = cmp.boot_diff(mats[a], mats[b], rng)
    est = (mats[a].mean(0) - mats[b].mean(0)).mean()
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    frac = np.mean(mats[a].mean(0) > mats[b].mean(0))
    pp = cmp.seed_permutation_p(np.median(mats[a], 1), np.median(mats[b], 1))
    return dict(est=est, lo=lo, hi=hi, p=p, frac=frac, perm=pp, na=len(mats[a]), nb=len(mats[b]))


def ptxt(p):
    return f"{p:.4f}" if p >= 1 / cmp.N_BOOT else f"<{1 / cmp.N_BOOT:.0e}"


def build_report(final_dir=FINAL_DIR, logs_dir=LOGS_DIR, allow_partial=False):
    """-> (lines, complete). complete is False if any input is missing."""
    out = []
    P = lambda s="": out.append(s)
    selection = load_selection(final_dir)
    arms = {n: load_arm(n, final_dir, selection) for n in ARMS}
    missing = list(dict.fromkeys(m for n in ARMS for m in arms[n]["missing"]))
    complete = not missing
    if not complete:
        P("INCOMPLETE - not the pre-registered result" if allow_partial
          else "INCOMPLETE - refusing to compute partial results (use --allow-partial for development)")
        for m in missing:
            P(f"  missing: {m}")
        P()
        if not allow_partial:
            return out, False
        P("EXPLORATORY / PARTIAL: only comparisons whose arms are complete are run.\n")

    sel_mats, keys, notes = matrices(arms, "sel")
    raw_mats, _, notes2 = matrices(arms, "raw")
    for n in dict.fromkeys(notes + notes2):
        P(n)
    ok = lambda n, f="sel": len(arms[n][f]) == len(SEEDS)

    P("Context-ablation family (protocol section 9), Holm over 2: " +
      "; ".join(f"{f}: {LABEL[a]} vs {LABEL[b]}" for f, a, b in FAMILY))
    P("Endpoint: final-split harvest of the val-selected checkpoint of each run")
    P(f"{len(keys)} yield-scored episodes per run\n")

    P("Selected checkpoints (selection.json, made on val only)")
    for n in ARMS:
        P(f"  {LABEL[n]:<10} {arms[n]['picks'] if arms[n]['picks'] else 'no selection'}")

    P(f"\n{'arm (selected ckpt)':<20}{'runs':>5}{'median g':>10}{'p25 g':>8}{'crash':>7}   per-run medians")
    for n in sorted([n for n in ARMS if n in sel_mats], key=lambda n: -np.median(sel_mats[n].mean(0))):
        X = sel_mats[n]
        tag = "" if len(X) == len(SEEDS) else f"   PARTIAL (seeds {sorted(arms[n]['sel'])})"
        P(f"{LABEL[n]:<20}{len(X):>5}{np.median(X.mean(0)):>10.2f}{np.percentile(X.mean(0), 25):>8.2f}"
          f"{np.mean(list(arms[n]['crash_sel'].values())):>7.0%}   {np.round(np.median(X, 1), 2).tolist()}{tag}")
    for n in ARMS:
        if n not in sel_mats:
            P(f"{LABEL[n]:<20}{0:>5}   no results")

    rng = np.random.default_rng(0)
    run = [(f, a, b) for f, a, b in FAMILY if ok(a) and ok(b)]
    skipped = [(f, a, b) for f, a, b in FAMILY if (f, a, b) not in run]
    res = [(f, a, b, boot_compare(sel_mats, a, b, rng)) for f, a, b in run]
    P("\nPre-registered comparisons (mean paired difference, hierarchical 95% CI, Holm-adjusted p over the family of 2)")
    if res:
        # a comparison that cannot be run counts as p=1 so the Holm multipliers stay those of a family of 2
        raw = np.array([r[3]["p"] for r in res] + [1.0] * len(skipped))
        adj = cmp.holm(raw)[:len(res)]
        for (f, a, b, r), pa in zip(res, adj):
            P(f"  {f}  {LABEL[a]} - {LABEL[b]}: {r['est']:+.2f} g [{r['lo']:+.2f}, {r['hi']:+.2f}]  "
              f"p_holm={ptxt(pa)}  {LABEL[a]} better in {r['frac']:.0%} of episodes")
    for f, a, b in skipped:
        P(f"  {f}  {LABEL[a]} - {LABEL[b]}: not run (arm incomplete)")
    for f, a, b, r in res:
        P(f"\n  {f} seed-level exact permutation test {LABEL[a]} vs {LABEL[b]} "
          f"({r['na']} v {r['nb']} runs, per-run medians): p={r['perm']:.4f}  [headline, section 8]")

    if allow_partial and not complete and ok("lructl") and ok("lstmctl"):
        r = boot_compare(sel_mats, "lructl", "lstmctl", rng)
        P("\nEXPLORATORY SANITY CHECK (development only, not a pre-registered comparison): "
          "controls only, selected checkpoints")
        P(f"  {LABEL['lructl']} - {LABEL['lstmctl']}: {r['est']:+.2f} g [{r['lo']:+.2f}, {r['hi']:+.2f}]  "
          f"p_boot(unadjusted)={ptxt(r['p'])}  {LABEL['lructl']} better in {r['frac']:.0%} of episodes; "
          f"seed-level exact permutation p={r['perm']:.4f}")

    P("\n" + "=" * 100)
    P("Secondary endpoints (unadjusted p-values, no Holm correction)")
    P("\nLRU control vs the new arms, selected checkpoints (bootstrap p unadjusted)")
    for f, a, b in SECONDARY:
        if ok(a) and ok(b):
            r = boot_compare(sel_mats, a, b, rng)
            P(f"  {LABEL[a]} - {LABEL[b]}: {r['est']:+.2f} g [{r['lo']:+.2f}, {r['hi']:+.2f}]  "
              f"p_boot={ptxt(r['p'])}  seed-level permutation p={r['perm']:.4f}")
        else:
            P(f"  {LABEL[a]} - {LABEL[b]}: not run (arm incomplete)")

    P("\nRaw final checkpoints (continuity with section 4), same comparisons")
    P(f"{'arm (final ckpt)':<20}{'runs':>5}{'median g':>10}{'crash':>7}   per-run medians")
    for n in ARMS:
        if n in raw_mats:
            X = raw_mats[n]
            P(f"{LABEL[n]:<20}{len(X):>5}{np.median(X.mean(0)):>10.2f}"
              f"{np.mean(list(arms[n]['crash_raw'].values())):>7.0%}   {np.round(np.median(X, 1), 2).tolist()}")
    for f, a, b in CONTINUITY:
        if ok(a, "raw") and ok(b, "raw"):
            r = boot_compare(raw_mats, a, b, rng)
            P(f"  {LABEL[a]} - {LABEL[b]}: {r['est']:+.2f} g [{r['lo']:+.2f}, {r['hi']:+.2f}]  "
              f"p_boot={ptxt(r['p'])}  seed-level permutation p={r['perm']:.4f}")
        else:
            P(f"  {LABEL[a]} - {LABEL[b]}: not run (arm incomplete)")

    P("\nCrash rate per seed (selected / final checkpoint)")
    for n in ARMS:
        if arms[n]["crash_sel"]:
            fm = lambda d: {s: f"{v:.1%}" for s, v in sorted(d.items())}
            P(f"  {LABEL[n]:<10} selected mean {np.mean(list(arms[n]['crash_sel'].values())):.1%} {fm(arms[n]['crash_sel'])}")
            if arms[n]["crash_raw"]:
                P(f"  {'':<10} final    mean {np.mean(list(arms[n]['crash_raw'].values())):.1%} {fm(arms[n]['crash_raw'])}")
    ca = {n: list(arms[n]["crash_sel"].values()) for n in ARMS}
    for f, a, b in FAMILY + SECONDARY:
        if len(ca[a]) == len(SEEDS) and len(ca[b]) == len(SEEDS):
            P(f"  crash (selected) {LABEL[a]} vs {LABEL[b]}: seed-level exact permutation p={sec.perm_p(ca[a], ca[b]):.4f}")

    P(f"\nCurriculum from training logs (steps censored at the {BUDGET:,} budget)")
    P(f"{'arm':<10}{'seed':>5}{'D1 at':>11}{'D2 at':>11}{'tier':>6}{'dem':>5}{'abort':>7}{'steps':>11}  status")
    by = {}
    for n in ARMS:
        by[n] = []
        for s in SEEDS:
            r = parse_run(logs_dir, n, s)
            if r is None:
                P(f"{LABEL[n]:<10}{s:>5}  missing: no session logs under {' or '.join(log_dirs(n, s))}")
                continue
            P(f"{LABEL[n]:<10}{s:>5}{cf.fmt_step(r['d1']):>11}{cf.fmt_step(r['d2']):>11}"
              f"{'D' + str(r['final_tier']):>6}{r['demotions']:>5}{'yes' if r['abort'] else 'no':>7}"
              f"{r['steps']:>11,}  {'complete' if r['finished'] else 'INCOMPLETE (excluded)'}")
            if r["finished"]:
                by[n].append(r)
    P("\nPer arm (finished seeds only)")
    P(f"{'arm':<10}{'seeds':>6}{'reach D1':>10}{'reach D2':>10}{'med steps D1':>15}{'med steps D2':>15}")
    fm_med = lambda v: (f"{np.median(v):,.0f}" if np.median(v) < BUDGET else f">={BUDGET:,}")
    for n in ARMS:
        rs = by[n]
        if rs:
            cs = lambda k: [r[k] if r[k] is not None else BUDGET for r in rs]
            P(f"{LABEL[n]:<10}{len(rs):>6}{sum(r['d1'] is not None for r in rs):>10}"
              f"{sum(r['d2'] is not None for r in rs):>10}{fm_med(cs('d1')):>15}{fm_med(cs('d2')):>15}")
    P("(medians use steps censored at the budget; '>=' means at least half the seeds never reached the tier)")
    for f, a, b in FAMILY + SECONDARY:
        A, B = by[a], by[b]
        if len(A) < 2 or len(B) < 2:
            continue
        P(f"\n{f} pair {LABEL[a]} vs {LABEL[b]} ({len(A)} v {len(B)} seeds)")
        for tier, key in (("D1", "d1"), ("D2", "d2")):
            ka, kb = sum(r[key] is not None for r in A), sum(r[key] is not None for r in B)
            xa = [r[key] if r[key] is not None else BUDGET for r in A]
            xb = [r[key] if r[key] is not None else BUDGET for r in B]
            P(f"  seeds reaching {tier}: {LABEL[a]} {ka}/{len(A)}, {LABEL[b]} {kb}/{len(B)}  "
              f"Fisher exact two-sided p={sec.fisher_two_sided(ka, len(A), kb, len(B)):.4f}; "
              f"steps to {tier} mean {np.mean(xa):,.0f} v {np.mean(xb):,.0f}, exact permutation p={sec.perm_p(xa, xb):.4f}")
    return out, complete


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help=f"also write ctx/{OUT_NAME}")
    ap.add_argument("--allow-partial", action="store_true", help="development only: run what is complete")
    ap.add_argument("--final-dir", default=FINAL_DIR)
    ap.add_argument("--logs-dir", default=LOGS_DIR)
    args = ap.parse_args(argv)
    lines, complete = build_report(args.final_dir, args.logs_dir, args.allow_partial)
    text = "\n".join(lines) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8"))
    if not complete and not args.allow_partial:
        return 1
    if args.write:
        d = os.path.join(args.final_dir, "ctx")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, OUT_NAME), "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
