"""
run_bc_clone_and_eval.py — runs bc/bc_pretrain.py fresh (clones the expert law into
the real RecurrentPPO policy, zero RL steps) and evaluates it at D0/D1/D2 with
diagnostics/held_out_sweep.py. Reproduces the earlier v19 BC-clone result
(model_data/BEST_bc_clone_D2_validated) from a clean run and extends it to all
three tiers.

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

    # Generate a fresh BC clone; copy its output out of model_data/bc_warmstart (shared,
    # live infra) into this experiment's own results/ so it's self-contained.
    bc_script = os.path.join(ROOT, "bc", "bc_pretrain.py")
    rc = run(
        [sys.executable, bc_script, "--episodes", "24", "--epochs", "10", "--critic-epochs", "20"],
        os.path.join(RESULTS_DIR, "bc_pretrain_fresh.log"),
    )
    if rc != 0:
        print("  bc_pretrain.py failed or aborted — see log above before continuing.")
        return

    import shutil
    src_dir = os.path.join(ROOT, "model_data", "bc_warmstart")
    dst_dir = os.path.join(RESULTS_DIR, "bc_clone")
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"  Copied fresh BC clone into {dst_dir}")

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
