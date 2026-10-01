"""Kaggle CPU session 'rl-v3-rtu-s2': TD3 rtu on physics v3, seed 2, pinned to tag rl-protocol-v2.
Exploratory arm (docs/reports/comparison_protocol.md section 6). See ../../run_rl.py."""
import os
import subprocess

os.environ["RL_CORE"] = "rtu"
os.environ["RL_ENV"] = '{"TD3_SEED": 2}'
subprocess.run("git clone --depth 1 -b rl-protocol-v2 https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_rl.py").read())
