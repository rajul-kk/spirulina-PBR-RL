"""Kaggle CPU session 'rl-4m-lstm60-s1': TD3 lstm on physics v3, seed 1, pinned to tag rl-protocol-v3.
4M follow-up (docs/reports/comparison_protocol.md section 7): fresh LSTM with hidden-state reset 60; phase 1 trains to 2M (TD3_STEPS is raised to 4000000 after the 2M result is fetched). See ../../run_rl.py."""
import os
import subprocess

os.environ["RL_CORE"] = "lstm"
os.environ["RL_ENV"] = '{"TD3_SEED": 1, "TD3_HIDDEN_RESET_INTERVAL": 60, "TD3_STEPS": 4000000}'
subprocess.run("git clone --depth 1 -b rl-protocol-v3 https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_rl.py").read())
