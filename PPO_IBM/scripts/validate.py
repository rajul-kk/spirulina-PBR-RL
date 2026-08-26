"""
validate.py — the mandatory independent check on any checkpoint, in one command.

WHY THIS EXISTS: no mastery claim in this project has ever been trustworthy without held-out
validation. v14 advanced to D2 with both in-training gates passing and then scored median
0.4mg against a 90mg gate. v17 did the same and failed at BOTH tiers. The in-training
deterministic eval uses a 15-episode rolling window; held_out_sweep.py uses 40 fresh seeds
including adversarial cold starts, and that difference has repeatedly been decisive.

It also runs the action trace, because the SHAPE of the harvest profile has diagnosed every
failure mode here: never-harvest (v4/v14), drift-up (v15), start-high-decay-to-zero (v16b),
over-harvest-early (v17). Aggregate numbers alone hid all four.

Usage (from the repo root):
    python scripts/validate.py --model model_data/best_det_checkpoint/recurrent_ppo_genetic_ibm
    python scripts/validate.py --model <path> --norm <path> --n 40 --seeds 0 1 2 3
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
LOG_DIR = os.path.join(ROOT, "logs", "validation")

GATES = {1: (60.0, 30.0, 0.008, 0.10), 2: (90.0, 50.0, 0.011, 0.08)}
BC_REF = "BC clone (no RL): harvest 109.4mg  p25 63.8  od 0.0191  crash 0.0%  -> passes D2"


def run(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=ENV, check=False)
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def parse_sweep(text):
    # Surface the specific failure that WILL recur: a checkpoint saved under a different
    # (full rationale: docs/decision_history.md#--scripts-validate-py-42)
    m = re.search(r"spaces must have the same shape: \((\d+),\) != \((\d+),\)", text)
    if m:
        return {"error": f"OBS-SPACE MISMATCH: checkpoint expects {m.group(1)} channels, "
                         f"current env provides {m.group(2)}. This checkpoint predates an "
                         f"observation change and cannot be evaluated against today's env — "
                         f"either check out the matching env revision or regenerate it."}
    if "Traceback" in text:
        last = [ln for ln in text.splitlines() if ln.strip()][-1][:160]
        return {"error": f"script failed: {last}"}

    def grab(pat):
        mm = re.search(pat, text)
        return float(mm.group(1)) if mm else None
    return {
        "harvest": grab(r"harvested_mg\s+median\s*:\s*([0-9.]+)"),
        "p25": grab(r"harvested_mg\s+median\s*:\s*[0-9.]+\s+p25:\s*([0-9.]+)"),
        "od": grab(r"time_avg_od\s+median\s*:\s*([0-9.]+)"),
        "crash": grab(r"crash_rate\s*:\s*([0-9.]+)%"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint path WITHOUT .zip")
    ap.add_argument("--norm", default=None, help="VecNormalize pkl (default: beside the model)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--skip-trace", action="store_true")
    args = ap.parse_args()

    model = args.model[:-4] if args.model.endswith(".zip") else args.model
    norm = args.norm or os.path.join(os.path.dirname(model), "recurrent_vec_normalize.pkl")
    if not os.path.isfile(norm):
        print(f"  ABORT: normalisation stats not found at {norm}. A model scored against another "
              f"run's obs_rms is meaningless — this is how several results in this project were "
              f"initially misread.")
        sys.exit(1)
    tag = os.path.basename(os.path.dirname(os.path.abspath(model))) or "model"

    print(f"── validating {model}")
    print(f"   norm: {norm}")
    print(f"   {BC_REF}\n")

    results = {}
    for diff in (1, 2):
        print(f"  held-out sweep D{diff} (n={args.n}) ...", flush=True)
        text = run([PY, "-u", os.path.join("diagnostics", "held_out_sweep.py"),
                    "--model", model, "--norm", norm,
                    "--difficulty", str(diff), "--n", str(args.n)],
                   os.path.join(LOG_DIR, f"{tag}_sweep_D{diff}.log"))
        results[diff] = parse_sweep(text)

    print(f"\n  {'tier':<6} {'harvest':>9} {'p25':>8} {'od':>9} {'crash':>7}   verdict")
    for diff in (1, 2):
        r, g = results[diff], GATES[diff]
        if r.get("error"):
            print(f"  D{diff:<5} {r['error']}")
            continue
        if r["harvest"] is None:
            print(f"  D{diff:<5} — could not parse sweep output; see log")
            continue
        ok = (r["harvest"] >= g[0] and r["p25"] >= g[1]
              and r["od"] >= g[2] and (r["crash"] / 100.0) <= g[3])
        marks = (f"h{'+' if r['harvest'] >= g[0] else '-'} "
                 f"p{'+' if r['p25'] >= g[1] else '-'} "
                 f"od{'+' if r['od'] >= g[2] else '-'} "
                 f"c{'+' if (r['crash']/100.0) <= g[3] else '-'}")
        print(f"  D{diff:<5} {r['harvest']:9.1f} {r['p25']:8.1f} {r['od']:9.4f} "
              f"{r['crash']:6.1f}%   {'PASS' if ok else 'FAIL'}  [{marks}]")
    print(f"\n  gates: D1 harvest>=60 p25>=30 od>=0.008 crash<=10% | "
          f"D2 harvest>=90 p25>=50 od>=0.011 crash<=8%")

    if not args.skip_trace:
        print(f"\n  action traces at D2 (harvest fraction per 600-step block) ...")
        for seed in args.seeds:
            text = run([PY, "-u", os.path.join("diagnostics", "test_actions.py"),
                        "--model", model, "--norm", norm,
                        "--difficulty", "2", "--interval", "600", "--seed", str(seed)],
                       os.path.join(LOG_DIR, f"{tag}_trace_seed{seed}.log"))
            fracs = [ln.split()[4] for ln in text.splitlines() if ln.strip().startswith("Harvest")]
            done = next((ln.strip() for ln in text.splitlines() if "Episode done" in ln), "?")
            print(f"    seed {seed}: {' '.join(fracs)}")
            print(f"             {done}")
        print("\n  SHAPE CHECK — the profile diagnoses the failure mode, not just the score:")
        print("    ~0 throughout            -> degenerate 'never harvest' (v4/v14)")
        print("    starts high, decays to 0 -> within-episode collapse (v16b)")
        print("    high early, plateaus     -> over-harvests before the culture establishes (v17)")
        print("    ~0 early, rising later   -> correct: build biomass, then harvest the surplus")

    print(f"\n  logs -> {os.path.relpath(LOG_DIR, ROOT)}")


if __name__ == "__main__":
    main()
