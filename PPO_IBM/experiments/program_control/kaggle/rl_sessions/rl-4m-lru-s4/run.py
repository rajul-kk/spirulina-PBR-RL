"""Kaggle CPU session 'rl-4m-lru-s4': TD3 lru on physics v3, seed 4, pinned to tag rl-protocol-v3.
4M follow-up (docs/reports/comparison_protocol.md section 7): resumes rl-v3-lru-s4 from its 2M state to 4M. See ../../run_rl.py."""
import os
import subprocess

os.environ["RL_CORE"] = "lru"
os.environ["RL_ENV"] = '{"TD3_SEED": 4, "TD3_STEPS": 4000000}'
subprocess.run("git clone --depth 1 -b rl-protocol-v3 https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_rl.py").read())
