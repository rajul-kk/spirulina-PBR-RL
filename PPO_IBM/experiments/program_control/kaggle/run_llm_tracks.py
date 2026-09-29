"""Kaggle job: the LLM-writer tracks of the program head-to-head (GPU or CPU-only sessions).

Clones the public repo, serves models with Ollama, runs the same evolve.py loop (same packet,
seed program, 16-candidate budget and search split as the local tracks) for each selected
model, then scores each track's best program on the held-out test split. Everything lands in
/kaggle/working/ (evolve/<run>/..., test/<run>_D2.json, logs), which `kaggle kernels output`
downloads.

Env (set by the per-session wrapper in kaggle/sessions/):
  TRACKS   comma-separated run names from MODELS (default: all that fit the hardware)
Resume: attach an earlier session's output as a kernel source; any evolve/<run>/ found under
/kaggle/input is copied in and the loop continues where it stopped. No new candidate is
started after DEADLINE_H hours, so a session never loses work to the 12 h limit.
"""
import glob
import json
import os
import shutil
import subprocess
import time
import urllib.request

T0 = time.time()
DEADLINE_H = 10.5
OUT = "/kaggle/working"
REPO = f"{OUT}/repo"
PC = f"{REPO}/PPO_IBM/experiments/program_control"
MODELS = {                       # run name: (ollama tag, min total VRAM in MiB; 0 = CPU is fine)
    "qwen3_8b": ("qwen3:8b", 0),
    "llama3_2_3b": ("llama3.2:3b", 0),
    "qwen2_5_coder_7b": ("qwen2.5-coder:7b", 0),
    "qwen2_5_coder_14b": ("qwen2.5-coder:14b", 12000),
    "qwen2_5_coder_32b": ("qwen2.5-coder:32b", 24000),
}


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def hours():
    return (time.time() - T0) / 3600


smi = subprocess.run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
                     shell=True, capture_output=True, text=True)
vram = sum(int(x) for x in smi.stdout.split()) if smi.returncode == 0 else 0
workers = max(1, os.cpu_count() or 2)
tracks = [t for t in os.environ.get("TRACKS", ",".join(MODELS)).split(",") if t]
SUFFIX = os.environ.get("RUN_SUFFIX", "")          # e.g. "_v3r2": separate archive per repeat
BRANCH = os.environ.get("REPO_BRANCH", "main")
print(f"VRAM {vram} MiB, {workers} CPUs, tracks {tracks}", flush=True)

if not os.path.isdir(REPO):
    sh(f"git clone --depth 1 -b {BRANCH} https://github.com/rajul-kk/spirulina-PBR-RL.git {REPO}")
sh(f"git -C {REPO} log -1 --format='repo at %h %s'")
sh("pip -q install gymnasium", check=False)
sh("apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y zstd >/dev/null 2>&1", check=False)
sh("curl -fsSL https://ollama.com/install.sh | sh")
server = subprocess.Popen(["ollama", "serve"], stdout=open(f"{OUT}/ollama_serve.log", "w"),
                          stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        break
    except Exception:
        time.sleep(2)

# Resume: bring in archives from an earlier session's output, if one is attached.
for prev in glob.glob("/kaggle/input/**/evolve/*/archive.jsonl", recursive=True):
    run = os.path.basename(os.path.dirname(prev))
    dst = f"{OUT}/evolve/{run}"
    if run in [t + SUFFIX for t in tracks] and not os.path.exists(dst):
        shutil.copytree(os.path.dirname(prev), dst)
        print(f"resumed {run} from {os.path.dirname(prev)}", flush=True)

deadline = T0 + DEADLINE_H * 3600
env = (f"EVOLVE_RUNS_DIR={OUT}/evolve EVOLVE_WORKERS={workers} "
       f"EVOLVE_DEADLINE_EPOCH={deadline} PYTHONUNBUFFERED=1")
os.makedirs(f"{OUT}/test", exist_ok=True)
summary = []
for track in tracks:
    tag, need_vram = MODELS[track]
    run = track + SUFFIX
    if time.time() > deadline:
        print(f"skip {run}: past the {DEADLINE_H} h deadline", flush=True)
        continue
    if vram < need_vram:
        print(f"skip {run}: needs {need_vram} MiB VRAM, have {vram}", flush=True)
        continue
    t_run = time.time()
    sh(f"ollama pull {tag}", check=False)
    sh(f"cd {PC} && {env} python evolve.py auto --run {run} --model {tag} --budget 16 "
       f"2>&1 | tee -a {OUT}/{run}.log", check=False)
    arch_path = f"{OUT}/evolve/{run}/archive.jsonl"
    if os.path.exists(arch_path):
        arch = [json.loads(l) for l in open(arch_path) if l.strip()]
        ok = [a for a in arch if "summary" in a]
        best = max(ok, key=lambda a: a["fitness"])
        done = len(arch) - 1 >= 16
        if done:   # score on the held-out split only once the track has used its full budget
            sh(f"cd {PC} && python harness.py {OUT}/evolve/{run}/{best['file']} --split test "
               f"--workers {workers} --out {OUT}/test/{run}_D2.json 2>&1 | tee -a {OUT}/{run}.log",
               check=False)
        summary.append({"run": run, "model": tag, "candidates": len(arch) - 1, "complete": done,
                        "best": best["file"], "search_fitness": best["fitness"],
                        "hours": round((time.time() - t_run) / 3600, 2), "vram_mib": vram})
    sh(f"ollama rm {tag}", check=False)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)

server.terminate()
shutil.rmtree(REPO, ignore_errors=True)   # keep the output small: only results are downloaded
print(json.dumps(summary, indent=1), flush=True)
print(f"done in {hours():.2f} h", flush=True)
