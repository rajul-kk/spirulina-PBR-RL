"""Kaggle session 'cpu-llama3-2-3b': tracks llama3_2_3b. See ../../run_llm_tracks.py."""
import os
import subprocess

os.environ["TRACKS"] = "llama3_2_3b"
subprocess.run("git clone --depth 1 https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_llm_tracks.py").read())
