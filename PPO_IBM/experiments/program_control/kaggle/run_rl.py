"""Kaggle GPU job: one TD3 tuning run on physics v3, resumable across 12 h sessions.

Env (set by the per-session wrapper in kaggle/rl_sessions/<name>/run.py):
  RL_CORE      lstm | lru
  RL_ENV       JSON dict of extra environment variables (TD3_BC_COEF, PBR_HARVEST_REWARD, ...)
  REPO_BRANCH  branch to clone
Resume: attach the previous session's output as a kernel source; its model_data/ is copied in
and training continues with --resume. Training is stopped at 11 h so the session can save its
output (TD3 checkpoints every 50k steps, so at most 50k steps are lost).
"""
import glob
import json
import os
import shutil
import subprocess
import time

OUT = "/kaggle/working"
REPO = f"{OUT}/repo"
PPO = f"{REPO}/PPO_IBM"
CORE = os.environ.get("RL_CORE", "lstm")
EXTRA = json.loads(os.environ.get("RL_ENV", "{}"))
BRANCH = os.environ.get("REPO_BRANCH", "main")
WALL_S = 11 * 3600


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
if not os.path.isdir(REPO):
    sh(f"git clone --depth 1 -b {BRANCH} https://github.com/rajul-kk/spirulina-PBR-RL.git {REPO}")
sh(f"git -C {REPO} log -1 --format='repo at %h %s'")
sh("pip -q install gymnasium tqdm", check=False)

resume = False
for prev in glob.glob("/kaggle/input/**/model_data", recursive=True):
    if glob.glob(f"{prev}/td3*training_state*.pkl"):
        shutil.copytree(prev, f"{PPO}/model_data", dirs_exist_ok=True)
        resume = True
        print(f"resuming from {prev}", flush=True)
        break
os.makedirs(f"{PPO}/logs", exist_ok=True)

env = dict(os.environ, PYTHONIOENCODING="utf-8", TD3_THREADS="4", TD3_HIDDEN_RESET_INTERVAL="600",
           TD3_STEPS="2000000", **{k: str(v) for k, v in EXTRA.items()})
script = "td3/TD3_lru.py" if CORE == "lru" else "td3/TD3.py"
cmd = ["python", "-u", script] + (["--resume"] if resume else [])
t0 = time.time()
with open(f"{OUT}/train.log", "a") as log:
    proc = subprocess.Popen(cmd, cwd=PPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    while proc.poll() is None and time.time() - t0 < WALL_S:
        time.sleep(30)
    if proc.poll() is None:
        print("wall-clock limit: stopping to save output; resume in a new session", flush=True)
        proc.terminate()
        proc.wait(timeout=120)

# Keep what a resume and the evaluation need: checkpoints, best checkpoints, training state.
os.makedirs(f"{OUT}/model_data", exist_ok=True)
for p in glob.glob(f"{PPO}/model_data/td3*"):
    dst = f"{OUT}/model_data/{os.path.basename(p)}"
    (shutil.copytree(p, dst, dirs_exist_ok=True) if os.path.isdir(p) else shutil.copy(p, dst))
shutil.rmtree(REPO, ignore_errors=True)
print(f"session done in {(time.time() - t0) / 3600:.2f} h", flush=True)
