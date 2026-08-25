"""
run_bc_clone_and_eval.py — orchestrates a fresh behaviour-cloning run (bc/bc_pretrain.py)
and evaluates the resulting NO-RL clone against the D0/D1/D2 curriculum gates with
diagnostics/held_out_sweep.py, on the exact held-out cold-start distribution.

This does NOT retrain the expert law (see expert_sweep.py for testing the law itself
directly). It answers a different question: does supervised cloning of that law into
the actual RecurrentPPO policy network (obs normalization, LSTM, MlpLstmPolicy) preserve
its gate-clearing performance, or does something get lost in the clone itself (before
any RL fine-tuning is even applied)?

Context: model_data/BEST_bc_clone_D2_validated (from an earlier v19 run) already passed
D2 held-out with no RL fine-tuning (harvest 109.4mg, p25 63.8mg, time_avg_od 0.0191,
0% crash — all comfortably above the D2 gate). This script reproduces that result from
a clean run so it's independently verified in this experiment folder rather than just
cited from an old log, and evaluates it at ALL THREE tiers (D0/D1/D2), not just D1/D2.

Usage (from repo root, PPO_IBM/):
    python experiments/bc_scaffold/scripts/run_bc_clone_and_eval.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BC_MODEL = os.path.join(ROOT, "experiments", "bc_scaffold", "results", "bc_clone", "recurrent_ppo_genetic_ibm")
BC_NORM = os.path.join(ROOT, "experiments", "bc_scaffold", "results", "bc_clone", "recurrent_vec_normalize.pkl")


def run(cmd, log_path):
    print(f"\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # bc_pretrain.py prints box-drawing chars; avoid
                                        # cp1252 UnicodeEncodeError when stdout is redirected
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env)
    print(f"  exit={proc.returncode}  log={log_path}")
    return proc.returncode


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(BC_MODEL), exist_ok=True)

    # 1. Generate a fresh BC clone (reuses bc/bc_pretrain.py's already-validated expert law).
    #    Redirect its default output paths into this experiment folder rather than clobbering
    #    model_data/bc_warmstart (that path is live infra other scripts resume from).
    bc_env = os.environ.copy()
    bc_script = os.path.join(ROOT, "bc", "bc_pretrain.py")
    rc = run(
        [sys.executable, bc_script, "--episodes", "24", "--epochs", "10", "--critic-epochs", "20"],
        os.path.join(RESULTS_DIR, "bc_pretrain_fresh.log"),
    )
    if rc != 0:
        print("  bc_pretrain.py failed or aborted (expert may have failed its own D1 gate check) "
              "— see log above before continuing.")
        return

    # bc_pretrain.py always writes to model_data/bc_warmstart; copy it into this experiment's
    # results/ so this folder is self-contained and doesn't silently depend on shared state.
    import shutil
    src_dir = os.path.join(ROOT, "model_data", "bc_warmstart")
    dst_dir = os.path.join(RESULTS_DIR, "bc_clone")
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"  Copied fresh BC clone into {dst_dir}")

    # 2. Evaluate the NO-RL clone at all three tiers with the project's own held-out sweep.
    sweep_script = os.path.join(ROOT, "diagnostics", "held_out_sweep.py")
    model_path = os.path.join(dst_dir, "recurrent_ppo_genetic_ibm")
    norm_path = os.path.join(dst_dir, "recurrent_vec_normalize.pkl")
    for difficulty in (0, 1, 2):
        run(
            [sys.executable, sweep_script, "--model", model_path, "--norm", norm_path,
             "--difficulty", str(difficulty), "--n", "40"],
            os.path.join(RESULTS_DIR, f"bc_clone_held_out_D{difficulty}.log"),
        )

    print("\nDone. See results/bc_clone_held_out_D{0,1,2}.log for the gate verdicts.")


if __name__ == "__main__":
    main()
