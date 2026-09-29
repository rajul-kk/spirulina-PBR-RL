"""Kaggle GPU session 'rl-v3-lstm-linear': TD3 lstm on physics v3, extra env {"PBR_HARVEST_REWARD": "linear"}. See ../../run_rl.py."""
import os
import subprocess

os.environ["RL_CORE"] = "lstm"
os.environ["RL_ENV"] = '{"PBR_HARVEST_REWARD": "linear"}'
os.environ["REPO_BRANCH"] = "worktree-agent-a87c1f2edcbd1556b"
subprocess.run("git clone --depth 1 -b worktree-agent-a87c1f2edcbd1556b https://github.com/rajul-kk/spirulina-PBR-RL.git /kaggle/working/repo",
               shell=True, check=True)
exec(open("/kaggle/working/repo/PPO_IBM/experiments/program_control/kaggle/run_rl.py").read())
