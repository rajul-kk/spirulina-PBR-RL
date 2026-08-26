"""
run_training.py — safe launcher for a curriculum training run.

Every bug this guards against actually happened in this project:

  * DUAL PROCESS (~20h of v16 invalidated). A launch reported a non-zero exit and was assumed
    dead; it wasn't. A second launch meant two processes writing the same log AND the same
    checkpoint_dir / state_path / norm_path, silently corrupting each other. The interleaved
    log looked like a curriculum state-machine bug and cost hours to diagnose.
      -> refuses to start if a recurrent_ppo process is already alive, and verifies exactly
         one startup banner appears after launch.
  * STALE AUTO-RESUME (v14). Bare `--resume` scans a shared, never-cleared checkpoint dir and
    picked up an unrelated older checkpoint while pairing it with the current run's state file.
      -> resume requires an explicit path; the launcher pairs norm+state from that same
         directory rather than leaving whatever happened to be in model_data/.
  * UNPAIRED NORM/STATE (nearly hit at v17). The trainer reads norm_path/state_path from
    model_data/, NOT from the warm-start folder, so `--resume warmstart/model.zip` would have
    loaded a BC actor against the previous run's normalisation statistics.
      -> pairing is explicit and verified before launch.
  * SILENT NON-LAUNCH. `(tasklist | grep -ci python) && python ...` never ran the trainer,
    because `grep -c` exits 1 on zero matches and `&&` short-circuited.
      -> the launcher checks the log for real startup output instead of trusting exit codes.
  * UNATTRIBUTABLE CONFIG CHANGES (v22 changed three things at once plus the seed, and its
    regression could not be assigned to any of them).
      -> every run writes a config snapshot and appends a row to a registry CSV.

Usage (ALWAYS from the repo root — relative paths resolve against the working directory):
    python scripts/run_training.py --tag v25_my_change
    python scripts/run_training.py --tag v25_bc --resume model_data/bc_warmstart/recurrent_ppo_genetic_ibm.zip
    python scripts/run_training.py --tag v25 --archive-prev v24_std_anneal_run5
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments")]

MODEL_DATA = os.path.join(ROOT, "model_data")
CKPT_DIR = os.path.join(MODEL_DATA, "recurrent_checkpoints")
STATE_PATH = os.path.join(MODEL_DATA, "recurrent_training_state.pkl")
NORM_PATH = os.path.join(MODEL_DATA, "recurrent_vec_normalize.pkl")
FINAL_MODEL = os.path.join(MODEL_DATA, "recurrent_ppo_genetic_ibm.zip")
BEST_DET = os.path.join(MODEL_DATA, "best_det_checkpoint")
LOG_DIR = os.path.join(ROOT, "logs")
REGISTRY = os.path.join(ROOT, "model_data", "runs_registry.csv")
TRAINER = os.path.join(ROOT, "training", "recurrent_ppo.py")


def live_training_pids():
    """PIDs of running trainer processes.

    Matches on the COMMAND LINE rather than on 'python' — an unrelated python process once
    blocked a wait loop for hours. But the pattern must be 'recurrent_ppo.py', NOT
    'recurrent_ppo': the saved model is named `recurrent_ppo_genetic_ibm`, so the looser
    pattern matches every diagnostic and validation process that references the checkpoint.
    That false positive is not harmless in either direction — it would make this launcher
    refuse to start while a read-only sweep was running, and a kill loop built on the same
    pattern terminated a validation run mid-sweep."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*recurrent_ppo.py*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception as exc:                      # noqa: BLE001 - advisory check only
        print(f"  [WARN] could not enumerate processes ({exc}); proceeding without that guard")
        return []


def snapshot_config():
    """Capture every constant that has mattered to an outcome in this project, so a run's
    configuration is recoverable from its archive instead of reconstructed from memory."""
    import curriculum_schedule as cs
    import entropy_schedule as es
    from genetic_env import GeneticPhotobioreactorEnv as Env

    cfg = {
        # Fix #24: gate mode and seed are part of a run's identity. Without the seed recorded,
        # (full rationale: docs/decision_history.md#--scripts-run_training-py-87)
        "gate_mode": os.environ.get("GATE_MODE", "dual"),
        "run_seed": int(os.environ.get("RUN_SEED", "0")),
        "env_debug": os.environ.get("ENV_DEBUG", ""),
        "total_training_steps": cs.TOTAL_TRAINING_STEPS,
        "chunk_steps": cs.CHUNK_STEPS,
        "advance_targets": {str(k): v for k, v in cs.ADVANCE_TARGETS.items()},
        "plateau_chunks": cs.PLATEAU_CHUNKS,
        "max_plateau_kicks_per_difficulty": cs.MAX_PLATEAU_KICKS_PER_DIFFICULTY,
        "capability_demotion_chunks": cs.CAPABILITY_DEMOTION_CHUNKS,
        "entropy_init": es.ENTROPY_INIT, "entropy_min": es.ENTROPY_MIN,
        "std_band_low": es.STD_BAND_LOW, "std_band_high": es.STD_BAND_HIGH,
        "std_anneal_start_frac": getattr(es, "STD_ANNEAL_START_FRAC", None),
        "std_anneal_end_frac": getattr(es, "STD_ANNEAL_END_FRAC", None),
        "std_anneal_final": getattr(es, "STD_ANNEAL_FINAL", None),
        "env_obs_dim": int(Env(max_cells=10, initial_cells=1).observation_space.shape[0]),
        "env_harvest_interval_steps": Env.HARVEST_INTERVAL_STEPS
        if hasattr(Env, "HARVEST_INTERVAL_STEPS") else None,
        "env_target_mg_per_event": Env.TARGET_MG_PER_EVENT,
        "env_light_fouling_coef": Env.LIGHT_FOULING_COEF,
        "env_turb_fouling_coef": Env.TURB_FOULING_COEF,
        "env_harvest_pump_error": Env.HARVEST_PUMP_ERROR,
        "env_use_episode_phase": Env.USE_EPISODE_PHASE,
    }
    return cfg


def archive_previous(tag):
    dest = os.path.join(MODEL_DATA, f"archive_{tag}")
    os.makedirs(dest, exist_ok=True)
    moved = []
    for src in (FINAL_MODEL, NORM_PATH, STATE_PATH):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
            moved.append(os.path.basename(src))
    if os.path.isdir(BEST_DET):
        d = os.path.join(dest, "best_det")
        if not os.path.exists(d):
            shutil.copytree(BEST_DET, d)
            moved.append("best_det/")
    print(f"  archived previous run -> {os.path.relpath(dest, ROOT)} ({', '.join(moved) or 'nothing'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="short run identifier, e.g. v25_kl_anchor")
    ap.add_argument("--resume", default=None,
                    help="explicit checkpoint .zip to resume from (never bare --resume)")
    ap.add_argument("--archive-prev", default=None,
                    help="tag to archive the CURRENT model_data artefacts under before clearing")
    ap.add_argument("--note", default="", help="free-text description recorded in the registry")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"── run_training: {args.tag} ──")

    pids = live_training_pids()
    if pids:
        print(f"  ABORT: recurrent_ppo already running (PIDs {pids}). Two concurrent runs share "
              f"checkpoint/state/norm paths and will corrupt each other — this invalidated ~20h "
              f"of v16. Stop it first.")
        sys.exit(1)
    print("  no existing training process ✓")

    cfg = snapshot_config()
    print(f"  config: gate={cfg['gate_mode']} seed={cfg['run_seed']} "
          f"obs_dim={cfg['env_obs_dim']} steps={cfg['total_training_steps']:,} "
          f"turb_foul={cfg['env_turb_fouling_coef']} pump_err={cfg['env_harvest_pump_error']} "
          f"ep_phase={cfg['env_use_episode_phase']} std_anneal_final={cfg['std_anneal_final']}")

    if args.dry_run:
        print("  --dry-run: nothing changed")
        return

    if args.archive_prev:
        archive_previous(args.archive_prev)

    # Reset the checkpoint dir by MOVING it aside, never deleting: overlapping step numbers
    # (full rationale: docs/decision_history.md#--scripts-run_training-py-166)
    if os.path.isdir(CKPT_DIR) and os.listdir(CKPT_DIR):
        aside = os.path.join(MODEL_DATA, f"recurrent_checkpoints_pre_{args.tag}")
        if not os.path.exists(aside):
            shutil.move(CKPT_DIR, aside)
            print(f"  checkpoints moved aside -> {os.path.basename(aside)}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"  ABORT: --resume path does not exist: {args.resume}")
            sys.exit(1)
        src_dir = os.path.dirname(os.path.abspath(args.resume))
        for base, dst in (("recurrent_vec_normalize.pkl", NORM_PATH),
                          ("recurrent_training_state.pkl", STATE_PATH)):
            cand = os.path.join(src_dir, base)
            if os.path.isfile(cand):
                shutil.copy2(cand, dst)
                print(f"  paired {base} from the resume directory ✓")
            else:
                print(f"  [WARN] {base} not found beside the checkpoint — the trainer will read "
                      f"whatever is in model_data/, which may belong to a DIFFERENT run.")
    else:
        for p in (STATE_PATH, NORM_PATH):
            if os.path.isfile(p):
                os.remove(p)
        if os.path.isdir(BEST_DET):
            shutil.rmtree(BEST_DET)
        print("  fresh start: cleared state, norm and best_det ✓")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"training_run_{args.tag}.log")
    cfg_path = os.path.join(LOG_DIR, f"config_{args.tag}.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump({"tag": args.tag, "resume": args.resume, "note": args.note,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg}, fh, indent=2)

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-u", TRAINER] + (["--resume", args.resume] if args.resume else [])
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    print(f"  launched PID {proc.pid} -> {os.path.relpath(log_path, ROOT)}")

    # Verify from the LOG, not from the exit code. A previous launch reported failure while
    # actually running, and that mistaken inference is what caused the dual-process incident.
    banner, deadline = None, time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"  ABORT: process exited early (code {proc.returncode}). Log tail:")
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                for line in fh.readlines()[-15:]:
                    print("    " + line.rstrip())
            sys.exit(1)
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        if "Resume Summary" in txt:
            banner = txt.count("Recurrent PPO Curriculum Training")
            break
        time.sleep(5)

    if banner is None:
        print("  [WARN] no startup banner within 300s — check the log manually")
    elif banner != 1:
        print(f"  ABORT: {banner} startup banners in the log — more than one process is writing "
              f"to it. Kill everything and relaunch.")
        sys.exit(1)
    else:
        print("  exactly one startup banner ✓ run is healthy")

    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    new = not os.path.isfile(REGISTRY)
    with open(REGISTRY, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["tag", "started", "pid", "resume", "gate_mode", "seed", "obs_dim",
                        "total_steps", "turb_foul", "pump_err", "ep_phase",
                        "std_anneal_final", "note",
                        "best_harvest", "best_p25", "best_od", "best_crash", "result"])
        w.writerow([args.tag, time.strftime("%Y-%m-%d %H:%M:%S"), proc.pid, args.resume or "",
                    cfg["gate_mode"], cfg["run_seed"],
                    cfg["env_obs_dim"], cfg["total_training_steps"],
                    cfg["env_turb_fouling_coef"], cfg["env_harvest_pump_error"],
                    cfg["env_use_episode_phase"], cfg["std_anneal_final"], args.note,
                    "", "", "", "", "RUNNING"])
    print(f"  registry row appended -> {os.path.relpath(REGISTRY, ROOT)}")
    print(f"\n  Fill in results when it finishes:  python scripts/finish_run.py --tag {args.tag}")


if __name__ == "__main__":
    main()
