"""Analysis of the 4M follow-up study (docs/reports/comparison_protocol.md section 7).

Pre-registered family, Holm-corrected over three (endpoint and statistics as section 4):
  F1  LRU-600 @4M  vs LSTM-600 @4M
  F2  LRU-600 @4M  vs LSTM-60  @4M
  F3  LSTM-60 @2M  vs LSTM-600 @2M
Endpoint: mean paired per-episode harvest difference over yield-scored episodes (initial culture
> 80 cells), hierarchical bootstrap 95% CI (compare.py's boot_diff: same draws, seed 0), Holm over
the three, % of episodes better, and a seed-level exact permutation test on per-run medians.
Secondary (unadjusted, printed after): 2M->4M change in each seed's median harvest, best-det
checkpoint medians, crash rates, seeds reaching D1/D2 and steps to D1/D2 by 4M (training logs).

Inputs (all under --final-dir, default results/final; logs under --logs-dir, default results):
  followup/td3_<arm>_<2m|4m>__s<seed>.json         arm in lru, lstm, lstm60
  followup/secondary/td3_<arm>_<steps>_best__s<seed>.json
  td3_lru__s<seed>.json / td3_lstm__s<seed>.json   the section 3 (2M) results
  LSTM-600 seed 2 hit the D0 capability abort before 2M, so its 2M result (td3_lstm__s2.json)
  counts as its 4M result (protocol section 7); this is reported when applied.
  Logs: rl_v3/rl-v3-<core>-s<seed> (first 2M) then rl_4m/<arm>-s<seed> (resumed to 4M); lstm60
  seeds only have rl_4m/lstm60-s<seed>.

If any family input is missing the report says so at the top, runs only the comparisons whose
arms are complete, and is NOT the pre-registered result. Nothing is written unless --write.

  python compare_followup.py            # print only
  python compare_followup.py --write    # also write followup/compare_followup.txt
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(HERE)
LOGS_DIR = os.path.dirname(FINAL_DIR)
SEEDS = [1, 2, 3, 4, 5]
BUDGET = 4_000_000
OUT_NAME = "compare_followup.txt"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse, never copy: bootstrap / permutation / Holm from compare.py, parse_run helpers from
# td3_secondary.py. Both keep their work under `if __name__ == "__main__"`, so importing is safe.
cmp = _load_module("pc_compare", os.path.join(FINAL_DIR, "compare.py"))
sec = _load_module("pc_td3_secondary", os.path.join(FINAL_DIR, "td3_secondary.py"))

# arm key -> (label, core in the 2M/section-3 filenames, followup arm name, steps tag, log core)
# Each arm lists, per seed, the result file and best-checkpoint file relative to final_dir.
ARM_ORDER = ["lru600_4m", "lstm600_4m", "lstm60_4m", "lstm60_2m", "lstm600_2m", "lru600_2m"]
LABEL = {"lru600_4m": "LRU-600@4M", "lstm600_4m": "LSTM-600@4M", "lstm60_4m": "LSTM-60@4M",
         "lstm60_2m": "LSTM-60@2M", "lstm600_2m": "LSTM-600@2M", "lru600_2m": "LRU-600@2M"}
FAMILY = [("F1", "lru600_4m", "lstm600_4m"), ("F2", "lru600_4m", "lstm60_4m"),
          ("F3", "lstm60_2m", "lstm600_2m")]
FAMILY_ARMS = ["lru600_4m", "lstm600_4m", "lstm60_4m", "lstm60_2m", "lstm600_2m"]
CARRY_SEED = 2  # LSTM-600: aborted before 2M, its 2M result counts as its 4M result


def result_paths(arm, seed):
    """(result json, best-checkpoint json, carried_over) relative to the final dir."""
    if arm == "lru600_4m":
        return (f"followup/td3_lru_4m__s{seed}.json", f"followup/secondary/td3_lru_4m_best__s{seed}.json", False)
    if arm == "lstm600_4m":
        if seed == CARRY_SEED:
            return (f"td3_lstm__s{seed}.json", f"secondary/td3_lstm_best__s{seed}.json", True)
        return (f"followup/td3_lstm_4m__s{seed}.json", f"followup/secondary/td3_lstm_4m_best__s{seed}.json", False)
    if arm in ("lstm60_4m", "lstm60_2m"):
        t = arm[-2:]
        return (f"followup/td3_lstm60_{t}__s{seed}.json", f"followup/secondary/td3_lstm60_{t}_best__s{seed}.json", False)
    if arm == "lstm600_2m":
        return (f"td3_lstm__s{seed}.json", f"secondary/td3_lstm_best__s{seed}.json", False)
    if arm == "lru600_2m":
        return (f"td3_lru__s{seed}.json", f"secondary/td3_lru_best__s{seed}.json", False)
    raise KeyError(arm)


# Training-log directories per arm (relative to the logs dir), in replay order.
def log_dirs(arm, seed):
    if arm == "lru":
        return [f"rl_v3/rl-v3-lru-s{seed}", f"rl_4m/lru-s{seed}"]
    if arm == "lstm":
        return [f"rl_v3/rl-v3-lstm-s{seed}", f"rl_4m/lstm-s{seed}"]
    if arm == "lstm60":
        return [f"rl_4m/lstm60-s{seed}"]
    raise KeyError(arm)


# ---------------------------------------------------------------------------------- results
def read_result(path):
    """-> (per-episode harvest g keyed by (seed, init_cells), yield-scored only; crash rate)."""
    eps = json.load(open(path))["episodes"]
    ep = {(e["seed"], e["init_cells"]): e["harvested_mg"] / 1000 for e in eps if e["init_cells"] > 80}
    return ep, float(np.mean([e["crashed"] for e in eps]))


def load_arms(final_dir):
    """-> {arm: {"runs": {seed: {key: g}}, "crash": {seed: rate}, "best": {seed: median g},
    "missing": [paths], "missing_best": [paths], "carried": [seeds]}}"""
    arms = {}
    for arm in ARM_ORDER:
        a = dict(runs={}, crash={}, best={}, missing=[], missing_best=[], carried=[])
        for s in SEEDS:
            rel, brel, carried = result_paths(arm, s)
            p = os.path.join(final_dir, rel)
            if os.path.exists(p):
                a["runs"][s], a["crash"][s] = read_result(p)
                if carried:
                    a["carried"].append(s)
            else:
                a["missing"].append(rel)
            bp = os.path.join(final_dir, brel)
            if os.path.exists(bp):
                be, _ = read_result(bp)
                a["best"][s] = float(np.median(list(be.values())))
            else:
                a["missing_best"].append(brel)
        arms[arm] = a
    return arms


def matrices(arms):
    """Per-arm (runs x episodes) matrices in g over the episodes common to every loaded run."""
    key_sets = [set(r) for a in arms.values() for r in a["runs"].values()]
    notes = []
    if not key_sets:
        return {}, [], notes
    keys = sorted(set.intersection(*key_sets))
    if any(len(k) != len(keys) for k in key_sets):
        notes.append(f"WARNING: runs do not share one episode set; using the {len(keys)} common episodes")
    mats = {name: np.array([[a["runs"][s][k] for k in keys] for s in sorted(a["runs"])])
            for name, a in arms.items() if a["runs"]}
    return mats, keys, notes


# ------------------------------------------------------------------------------------- logs
def session_files(run_dir):
    """Session logs of one run directory in session-number order."""
    return sorted(glob.glob(os.path.join(run_dir, "session*", "train_summary.log")),
                  key=lambda p: int(re.search(r"session(\d+)", p).group(1)))


def replay_logs(paths):
    """Replay an ordered list of session logs. Same logic as td3_secondary.parse_run (which only
    accepts one run dir and re-sorts by session number, so it cannot join rl_v3 and rl_4m); the
    test checks the two agree on a single dir."""
    num = sec.num
    step, events, abort, complete = 0, [], False, False
    for path in paths:
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


def parse_followup_run(logs_dir, arm, seed):
    """Replay rl_v3 (first 2M) then rl_4m sessions. None if there are no logs at all."""
    paths = [p for d in log_dirs(arm, seed) for p in session_files(os.path.join(logs_dir, d))]
    if not paths:
        return None
    r = replay_logs(paths)
    for k in ("d1", "d2"):  # censor at the 4M budget
        if r[k] is not None and r[k] > BUDGET:
            r[k] = None
    # finished = ran to the budget, or ended by the capability abort (counts as finished, sec. 3)
    r["finished"] = r["complete"] and (r["steps"] >= BUDGET or r["abort"])
    return r


# ---------------------------------------------------------------------------------- report
def fmt_step(s):
    return f"{s:,}" if s is not None else "-"


def fmt_med(vals):
    v = np.median(vals)
    return f"{v:,.0f}" if v < BUDGET else f">={BUDGET:,}"


def build_report(final_dir=FINAL_DIR, logs_dir=LOGS_DIR):
    """-> (lines, complete: bool). complete is False if any family input is missing."""
    out = []
    P = lambda s="": out.append(s)

    arms = load_arms(final_dir)
    mats, keys, notes = matrices(arms)
    fam_missing = [m for n in FAMILY_ARMS for m in arms[n]["missing"]]
    fam_missing = list(dict.fromkeys(fam_missing))  # lstm600_4m/2m share files only via carry-over
    complete = not fam_missing

    if not complete:
        P("INCOMPLETE - not the pre-registered result")
        for m in fam_missing:
            P(f"  missing: {m}")
        P()
    for n in notes:
        P(n)
    if arms["lstm600_4m"]["carried"]:
        P(f"Carry-over applied: LSTM-600 seed {', '.join(map(str, arms['lstm600_4m']['carried']))} "
          f"(aborted before 2M); its 2M result ({result_paths('lstm600_4m', CARRY_SEED)[0]}) is scored as its 4M result.")
        P()
    P("Follow-up family (protocol section 7), Holm over 3: " +
      "; ".join(f"{f}: {LABEL[a]} vs {LABEL[b]}" for f, a, b in FAMILY))
    P(f"{len(keys)} yield-scored episodes per run\n")

    # ---- arm table
    P(f"{'arm':<13}{'runs':>5}{'median g':>10}{'p25 g':>8}{'crash':>7}   per-run medians")
    shown = [n for n in ARM_ORDER if n in mats]
    for n in sorted(shown, key=lambda n: -np.median(mats[n].mean(0))):
        X, a = mats[n], arms[n]
        med = np.median(X, axis=1)
        tag = "" if len(X) == len(SEEDS) else f"   PARTIAL (seeds {sorted(a['runs'])})"
        P(f"{LABEL[n]:<13}{len(X):>5}{np.median(X.mean(0)):>10.2f}{np.percentile(X.mean(0), 25):>8.2f}"
          f"{np.mean(list(a['crash'].values())):>7.0%}   {np.round(med, 2).tolist()}{tag}")
    for n in ARM_ORDER:
        if n not in mats:
            P(f"{LABEL[n]:<13}{0:>5}   no results")

    # ---- pre-registered comparisons
    runnable = [(f, a, b) for f, a, b in FAMILY if not arms[a]["missing"] and not arms[b]["missing"]]
    skipped = [(f, a, b) for f, a, b in FAMILY if (f, a, b) not in runnable]
    rng = np.random.default_rng(0)
    res = []
    for f, a, b in runnable:
        d = cmp.boot_diff(mats[a], mats[b], rng)
        est = (mats[a].mean(0) - mats[b].mean(0)).mean()
        p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        res.append((f, a, b, est, *np.percentile(d, [2.5, 97.5]), p))
    P("\nPre-registered comparisons (mean paired difference, hierarchical 95% CI, Holm-adjusted p over the family of 3)")
    if res:
        # a comparison that cannot be run counts as p=1 so the Holm multipliers stay those of a family of 3
        raw = np.array([r[-1] for r in res] + [1.0] * len(skipped))
        adj = cmp.holm(raw)[:len(res)]
        for (f, a, b, est, lo, hi, p), pa in zip(res, adj):
            frac = np.mean(mats[a].mean(0) > mats[b].mean(0))
            ptxt = f"{pa:.4f}" if pa >= 1 / cmp.N_BOOT else f"<{1 / cmp.N_BOOT:.0e}"
            P(f"  {f}  {LABEL[a]} - {LABEL[b]}: {est:+.2f} g [{lo:+.2f}, {hi:+.2f}]  p_holm={ptxt}  "
              f"{LABEL[a]} better in {frac:.0%} of episodes")
    for f, a, b in skipped:
        P(f"  {f}  {LABEL[a]} - {LABEL[b]}: not run (arm incomplete)")
    for f, a, b in runnable:
        pp = cmp.seed_permutation_p(np.median(mats[a], 1), np.median(mats[b], 1))
        P(f"\n  {f} seed-level exact permutation test {LABEL[a]} vs {LABEL[b]} "
          f"({len(mats[a])} v {len(mats[b])} runs, per-run medians): p={pp:.4f}")

    # ---- secondary
    P("\n" + "=" * 100)
    P("Secondary endpoints (unadjusted p-values, no Holm correction)")
    sec_missing = [m for n in ARM_ORDER for m in arms[n]["missing"] if n not in FAMILY_ARMS]
    sec_missing += [m for n in ARM_ORDER for m in arms[n]["missing_best"]]
    if sec_missing:
        P("Missing secondary inputs:")
        for m in dict.fromkeys(sec_missing):
            P(f"  missing: {m}")

    P("\n2M -> 4M change in each seed's median harvest (g, yield-scored episodes, final checkpoint)")
    P(f"{'arm':<10}{'seed':>5}{'2M':>9}{'4M':>9}{'change':>9}")
    for label, a2, a4 in (("LRU-600", "lru600_2m", "lru600_4m"), ("LSTM-600", "lstm600_2m", "lstm600_4m"),
                          ("LSTM-60", "lstm60_2m", "lstm60_4m")):
        ds = []
        for s in SEEDS:
            r2, r4 = arms[a2]["runs"].get(s), arms[a4]["runs"].get(s)
            m2 = np.median(list(r2.values())) if r2 else None
            m4 = np.median(list(r4.values())) if r4 else None
            f = lambda v: f"{v:.2f}" if v is not None else "-"
            ch = m4 - m2 if m2 is not None and m4 is not None else None
            ds.append(ch)
            P(f"{label:<10}{s:>5}{f(m2):>9}{f(m4):>9}{(f'{ch:+.2f}' if ch is not None else '-'):>9}"
              + ("   (carried over: 2M result is the 4M result)" if s in arms[a4]["carried"] else ""))
        ds = [d for d in ds if d is not None]
        if ds:
            P(f"{label:<10}{'mean':>5}{'':>9}{'':>9}{np.mean(ds):>+9.2f}   ({sum(d > 0 for d in ds)}/{len(ds)} seeds up)")

    P("\nBest-det checkpoint: per-seed median harvest (g, yield-scored episodes)")
    for n in ARM_ORDER:
        b = arms[n]["best"]
        if b:
            P(f"  {LABEL[n]:<13} median of seeds {np.median(list(b.values())):.2f}   "
              f"{ {s: round(v, 2) for s, v in sorted(b.items())} }")
        else:
            P(f"  {LABEL[n]:<13} no results")

    P("\nCrash rate of the final checkpoint (per seed)")
    for n in ARM_ORDER:
        c = arms[n]["crash"]
        if c:
            P(f"  {LABEL[n]:<13} mean {np.mean(list(c.values())):.1%}   "
              f"{ {s: f'{v:.1%}' for s, v in sorted(c.items())} }")

    # ---- curriculum from logs
    P("\nCurriculum from training logs (steps censored at the 4,000,000 budget)")
    P(f"{'arm':<10}{'seed':>5}{'D1 at':>11}{'D2 at':>11}{'tier':>6}{'dem':>5}{'abort':>7}{'steps':>11}  status")
    logs = {}
    for core, label in (("lru", "LRU-600"), ("lstm", "LSTM-600"), ("lstm60", "LSTM-60")):
        for s in SEEDS:
            r = parse_followup_run(logs_dir, core, s)
            logs[(core, s)] = r
            if r is None:
                P(f"{label:<10}{s:>5}  missing: no session logs under {' or '.join(log_dirs(core, s))}")
                continue
            P(f"{label:<10}{s:>5}{fmt_step(r['d1']):>11}{fmt_step(r['d2']):>11}{'D' + str(r['final_tier']):>6}"
              f"{r['demotions']:>5}{'yes' if r['abort'] else 'no':>7}{r['steps']:>11,}  "
              f"{'complete' if r['finished'] else 'INCOMPLETE (excluded)'}")
    by = {c: [logs[(c, s)] for s in SEEDS if logs[(c, s)] and logs[(c, s)]["finished"]]
          for c in ("lru", "lstm", "lstm60")}
    P("\nPer arm (finished seeds only)")
    P(f"{'arm':<10}{'seeds':>6}{'reach D1':>10}{'reach D2':>10}{'med steps D1':>15}{'med steps D2':>15}")
    for c, label in (("lru", "LRU-600"), ("lstm", "LSTM-600"), ("lstm60", "LSTM-60")):
        rs = by[c]
        if not rs:
            continue
        cs = lambda k: [r[k] if r[k] is not None else BUDGET for r in rs]
        P(f"{label:<10}{len(rs):>6}{sum(r['d1'] is not None for r in rs):>10}{sum(r['d2'] is not None for r in rs):>10}"
          f"{fmt_med(cs('d1')):>15}{fmt_med(cs('d2')):>15}")
    P("(medians use steps censored at the budget; '>=' means at least half the seeds never reached the tier)")

    for tag, ca, cb, na, nb in (("F1", "lru", "lstm", "lru600_4m", "lstm600_4m"),
                                ("F2", "lru", "lstm60", "lru600_4m", "lstm60_4m")):
        A, B = by[ca], by[cb]
        if len(A) < 2 or len(B) < 2:
            continue
        la, lb = LABEL[na].split("@")[0], LABEL[nb].split("@")[0]
        P(f"\n{tag} pair {la} vs {lb} at 4M ({len(A)} v {len(B)} seeds)")
        for tier, key in (("D1", "d1"), ("D2", "d2")):
            ka, kb = sum(r[key] is not None for r in A), sum(r[key] is not None for r in B)
            P(f"  seeds reaching {tier}: {la} {ka}/{len(A)}, {lb} {kb}/{len(B)}  "
              f"Fisher exact two-sided p={sec.fisher_two_sided(ka, len(A), kb, len(B)):.4f}")
        for tier, key in (("D1", "d1"), ("D2", "d2")):
            xa = [r[key] if r[key] is not None else BUDGET for r in A]
            xb = [r[key] if r[key] is not None else BUDGET for r in B]
            P(f"  steps to {tier} (censored at budget): mean {la} {np.mean(xa):,.0f}, {lb} {np.mean(xb):,.0f}; "
              f"median {la} {fmt_med(xa)}, {lb} {fmt_med(xb)}; exact permutation p={sec.perm_p(xa, xb):.4f}")
        ca_, cb_ = list(arms[na]["crash"].values()), list(arms[nb]["crash"].values())
        if len(ca_) == len(SEEDS) and len(cb_) == len(SEEDS):
            P(f"  final-checkpoint crash rate: mean {la} {np.mean(ca_):.1%}, {lb} {np.mean(cb_):.1%}; "
              f"seed-level exact permutation p={sec.perm_p(ca_, cb_):.4f}")
        else:
            P("  final-checkpoint crash rate: missing result files, test skipped")
    return out, complete


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help=f"also write followup/{OUT_NAME}")
    ap.add_argument("--final-dir", default=FINAL_DIR, help="results/final directory")
    ap.add_argument("--logs-dir", default=LOGS_DIR, help="results directory holding rl_v3 and rl_4m")
    args = ap.parse_args(argv)
    lines, complete = build_report(args.final_dir, args.logs_dir)
    text = "\n".join(lines) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8"))
    if args.write:
        d = os.path.join(args.final_dir, "followup")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, OUT_NAME), "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
