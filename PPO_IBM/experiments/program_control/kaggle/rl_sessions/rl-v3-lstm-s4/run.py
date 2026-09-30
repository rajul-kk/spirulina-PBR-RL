"""Kaggle CPU session 'rl-v3-lstm-s4': TD3 lstm on physics v3, seed 4, pinned to tag rl-protocol-v1.
Part of the LRU/LSTM seed sweep in docs/reports/comparison_protocol.md. See ../../run_rl.py."""
import os
import subprocess

os.environ["RL_CORE"] = "lstm"
os.environ["RL_ENV"] = '{"TD3_SEED": 4}'
subprocess.run("git clone --depth 1 -b rl-protocol-v1 https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_rl.py").read())
