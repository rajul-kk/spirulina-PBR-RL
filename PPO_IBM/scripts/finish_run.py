"""
finish_run.py — close out a training run: read its best deterministic checkpoint, score it
against the curriculum gates, and write the result back into the run registry.

WHY: comparing runs in this project meant grepping five multi-megabyte logs by hand and
holding the numbers in working memory. That is how v21's od 0.0094 came to be treated as a
reproducible level for a while — it was the top of a 0.0054-0.0094 spread, and nothing made
the spread visible. A registry with one row per run makes that mistake hard to repeat.

Usage (from the repo root):
    python scripts/finish_run.py --tag v24_std_anneal_run5
    python scripts/finish_run.py --tag v24_std_anneal_run5 --result "no D2; noise-dependent"
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments")]

MODEL_DATA = os.path.join(ROOT, "model_data")
BEST_DET_INFO = os.path.join(MODEL_DATA, "best_det_checkpoint", "best_det_info.txt")
REGISTRY = os.path.join(MODEL_DATA, "runs_registry.csv")

# The behaviour-cloned controller: the only artefact in this project that passes held-out D2,
# (full rationale: docs/decision_history.md#--scripts-finish_run-py-26)
BC_REF = {"harvest": 109.4, "p25": 63.8, "od": 0.0191, "crash": 0.0}


def read_best_det():
    if not os.path.isfile(BEST_DET_INFO):
        return None
    out = {}
    with open(BEST_DET_INFO, encoding="utf-8") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                out[k] = v
    return out


def gate_verdict(h, p25, od, crash):
    import curriculum_schedule as cs
    lines = []
    for tier in (0, 1, 2):
        t = cs.ADVANCE_TARGETS[tier]
        checks = [
            ("harvest", h, t["min_median_harvested_mg"], h >= t["min_median_harvested_mg"]),
            ("p25", p25, t["min_p25_harvested_mg"], p25 >= t["min_p25_harvested_mg"]),
            ("od", od, t["min_median_time_avg_od"], od >= t["min_median_time_avg_od"]),
            ("crash", crash, t["max_crash_rate"], crash <= t["max_crash_rate"]),
        ]
        ok = all(c[3] for c in checks)
        detail = "  ".join(f"[{'+' if c[3] else '-'}]{c[0]}:{c[1]:.4g}/{c[2]:.4g}" for c in checks)
        lines.append(f"    D{tier}: {'PASS' if ok else 'FAIL'}   {detail}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--result", default="", help="one-line verdict for the registry")
    args = ap.parse_args()

    info = read_best_det()
    if info is None:
        print(f"  no best_det_checkpoint found at {os.path.relpath(BEST_DET_INFO, ROOT)}")
        sys.exit(1)

    h = float(info.get("median_harvested_mg", 0.0))
    p25 = float(info.get("p25_harvested_mg", 0.0))
    od = float(info.get("median_time_avg_od", 0.0))
    crash = float(info.get("crash_rate", 0.0))

    print(f"── {args.tag} ──")
    # info stores difficulty already prefixed ("D0"), so don't add another "D".
    print(f"  best deterministic policy: step {info.get('step','?')} chunk {info.get('chunk','?')} "
          f"({info.get('difficulty','?')})")
    print(f"    harvest={h:.1f}mg  p25={p25:.1f}  time_avg_od={od:.4f}  crash={crash:.1%}")
    print(f"  BC reference (no RL):  harvest={BC_REF['harvest']}mg  p25={BC_REF['p25']}  "
          f"od={BC_REF['od']}  crash=0.0%")
    print(f"    -> {'BEATS' if od > BC_REF['od'] else 'below'} the BC controller on time_avg_od")
    print("  gate scorecard (best deterministic policy vs each tier):")
    for line in gate_verdict(h, p25, od, crash):
        print(line)
    print("\n  NOTE: this is the IN-TRAINING deterministic eval (15-episode window). It is not "
          "a substitute for held-out validation — v14 and v17 both passed the in-training gates "
          "and then failed held_out_sweep.py. Run: python scripts/validate.py --model <path>")

    if not os.path.isfile(REGISTRY):
        print(f"\n  no registry at {os.path.relpath(REGISTRY, ROOT)}; nothing to update")
        return
    with open(REGISTRY, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, updated = rows[0], 0
    idx = {name: i for i, name in enumerate(header)}
    for row in rows[1:]:
        if row and row[idx["tag"]] == args.tag:
            for key, val in (("best_harvest", f"{h:.1f}"), ("best_p25", f"{p25:.1f}"),
                             ("best_od", f"{od:.4f}"), ("best_crash", f"{crash:.4f}"),
                             ("result", args.result or "completed")):
                if key in idx:
                    while len(row) <= idx[key]:
                        row.append("")
                    row[idx[key]] = val
            updated += 1
    if updated:
        with open(REGISTRY, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        print(f"\n  registry updated ({updated} row) -> {os.path.relpath(REGISTRY, ROOT)}")
    else:
        print(f"\n  no registry row tagged '{args.tag}' — was it launched via run_training.py?")


if __name__ == "__main__":
    main()
