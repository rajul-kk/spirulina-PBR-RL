"""Kaggle session 'gpu-coder-14b-32b': tracks qwen2_5_coder_14b,qwen2_5_coder_32b. See ../../run_llm_tracks.py."""
import os
import subprocess

os.environ["TRACKS"] = "qwen2_5_coder_14b,qwen2_5_coder_32b"
subprocess.run("git clone --depth 1 https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_llm_tracks.py").read())
