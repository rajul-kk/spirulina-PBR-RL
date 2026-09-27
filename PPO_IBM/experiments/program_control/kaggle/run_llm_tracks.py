"""Kaggle GPU job: the LLM-writer tracks of the program head-to-head.

Clones the public repo, serves models with Ollama on the GPU, runs the same evolve.py loop
(same packet, seed program, 16-candidate budget and search split as the local tracks) for each
model, then scores each track's best program on the held-out test split. Everything lands in
/kaggle/working/ (evolve/<run>/..., test/<run>_D2.json, logs), which `kaggle kernels output`
downloads.
"""
import glob
import json
import os
import subprocess
import time
import urllib.request

T0 = time.time()
BUDGET_S = 10.0 * 3600          # stop starting new tracks after this (sessions end at 12 h)
OUT = "/kaggle/working"
REPO = f"{OUT}/repo"
PC = f"{REPO}/PPO_IBM/experiments/program_control"
MODELS = [                       # (run name, ollama tag, min total VRAM in MiB)
    ("qwen3_8b", "qwen3:8b", 0),
    ("llama3_2_3b", "llama3.2:3b", 0),
    ("qwen2_5_coder_14b", "qwen2.5-coder:14b", 12000),
    ("qwen2_5_coder_32b", "qwen2.5-coder:32b", 24000),
]


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def hours():
    return (time.time() - T0) / 3600


sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
vram = sum(int(x) for x in subprocess.run(
    "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", shell=True,
    capture_output=True, text=True).stdout.split())
workers = max(1, os.cpu_count() or 2)
print(f"total VRAM {vram} MiB, {workers} CPUs", flush=True)

sh(f"git clone --depth 1 https://github.com/rajul-kk/spirulina-PBR-RL.git {REPO}")
sh("git -C " + REPO + " log -1 --format='repo at %h %s'")
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

env = f"EVOLVE_RUNS_DIR={OUT}/evolve EVOLVE_WORKERS={workers} PYTHONUNBUFFERED=1"
os.makedirs(f"{OUT}/test", exist_ok=True)
summary = []
for run, tag, need_vram in MODELS:
    if hours() * 3600 > BUDGET_S:
        print(f"skip {run}: {hours():.1f} h used", flush=True)
        continue
    if vram < need_vram:
        print(f"skip {run}: needs {need_vram} MiB VRAM, have {vram}", flush=True)
        continue
    t_run = time.time()
    sh(f"ollama pull {tag}", check=False)
    sh(f"cd {PC} && {env} python evolve.py auto --run {run} --model {tag} --budget 16 "
       f"2>&1 | tee {OUT}/{run}.log", check=False)
    arch_path = f"{OUT}/evolve/{run}/archive.jsonl"
    if os.path.exists(arch_path):
        arch = [json.loads(l) for l in open(arch_path) if l.strip()]
        ok = [a for a in arch if "summary" in a]
        best = max(ok, key=lambda a: a["fitness"])
        best_file = f"{OUT}/evolve/{run}/{best['file']}"
        sh(f"cd {PC} && python harness.py {best_file} --split test --workers {workers} "
           f"--out {OUT}/test/{run}_D2.json 2>&1 | tee -a {OUT}/{run}.log", check=False)
        summary.append({"run": run, "model": tag, "candidates": len(arch) - 1, "best": best["file"],
                        "search_fitness": best["fitness"], "hours": round((time.time() - t_run) / 3600, 2)})
    sh(f"ollama rm {tag}", check=False)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)

server.terminate()
sh(f"rm -rf {REPO}", check=False)        # keep the output small: only results are downloaded
print(json.dumps(summary, indent=1), flush=True)
print(f"done in {hours():.2f} h", flush=True)
