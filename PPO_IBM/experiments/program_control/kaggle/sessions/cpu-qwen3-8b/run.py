"""Kaggle session 'cpu-qwen3-8b': tracks qwen3_8b. See ../../run_llm_tracks.py."""
import os
import subprocess

os.environ["TRACKS"] = "qwen3_8b"
subprocess.run("git clone --depth 1 https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_llm_tracks.py").read())
