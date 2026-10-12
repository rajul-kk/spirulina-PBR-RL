"""Kaggle CPU session 'rl-ctx-w150-s4': TD3 lstm on physics v3, seed 4, pinned to tag rl-protocol-v4.
Training-context ablation (docs/reports/comparison_protocol.md section 9): W150-LSTM (150-step replay windows), 2M steps, hidden-state reset 600 (run_rl.py default). See ../../run_rl.py."""
import os
import subprocess

os.environ["RL_CORE"] = "lstm"
os.environ["RL_ENV"] = '{"TD3_SEED": 4, "TD3_STEPS": 2000000, "TD3_SEQ_LEN": 150}'
subprocess.run("git clone --depth 1 -b rl-protocol-v4 https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_rl.py").read())
