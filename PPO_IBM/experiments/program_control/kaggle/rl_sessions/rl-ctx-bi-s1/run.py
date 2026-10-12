"""Kaggle CPU session 'rl-ctx-bi-s1': TD3 lstm on physics v3, seed 1, pinned to tag rl-protocol-v4.
Training-context ablation (docs/reports/comparison_protocol.md section 9): BI-LSTM (burn-in 40, state stride 10), 2M steps, hidden-state reset 600 (run_rl.py default). See ../../run_rl.py."""
import os
import subprocess

os.environ["RL_CORE"] = "lstm"
os.environ["RL_ENV"] = '{"TD3_SEED": 1, "TD3_STEPS": 2000000, "TD3_BURN_IN": 40, "TD3_STATE_STRIDE": 10}'
subprocess.run("git clone --depth 1 -b rl-protocol-v4 https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_rl.py").read())
