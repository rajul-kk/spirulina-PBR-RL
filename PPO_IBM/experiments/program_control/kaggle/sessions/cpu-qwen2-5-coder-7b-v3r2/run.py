"""Kaggle session 'cpu-qwen2-5-coder-7b-v3r2': track qwen2_5_coder_7b on physics v3, repeat 2. See ../../run_llm_tracks.py."""
import os
import subprocess

os.environ["TRACKS"] = "qwen2_5_coder_7b"
os.environ["RUN_SUFFIX"] = "_v3r2"
os.environ["REPO_BRANCH"] = "worktree-agent-a87c1f2edcbd1556b"
subprocess.run("git clone --depth 1 -b worktree-agent-a87c1f2edcbd1556b https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_llm_tracks.py").read())
