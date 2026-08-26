
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-evaluate_agent-py-2)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import torch
import sys
import os
import argparse
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import pandas as pd

# Add environments to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'environments'))

sys.path.insert(0, os.path.dirname(__file__))

from heavy_env import HeavyPhotobioreactorEnv
from genetic_env import GeneticPhotobioreactorEnv
from light_env import LightPhotobioreactorEnv
from total_env import TotalPhotobioreactorEnv
from TD_MPC2 import TDMPC2Agent as TDMPC2AgentClass, OBS_DIM as TDMPC2_OBS_DIM
from Var_MPC import TDMPC2Agent as VarMPCAgentClass, OBS_DIM as VARMPC_OBS_DIM
from recurrent_ppo import ActionSmoothnessWrapper


class RandomAgent:
    """Baseline agent that takes a smoothed random walk to avoid L2 penalties."""
    def __init__(self, action_space):
        self.action_space = action_space
        self.current_action = np.zeros(action_space.shape, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        """Returns a smoothly drifting random action."""
        delta = np.random.normal(0, 0.05, size=self.action_space.shape).astype(np.float32)
        self.current_action = np.clip(self.current_action + delta, -1.0, 1.0)
        return self.current_action, None


class Evaluator:
    def __init__(self, env_name="heavy", ppo_model_path=None, enable_ppo=True, enable_tdmpc2=True, enable_varmpc=True,
                 enable_sac=True, enable_random=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths
        base_dir = "model_data"
        recurrent_ppo_path = ppo_model_path if ppo_model_path else f"{base_dir}/PPO_heavy_final"
        sac_path = f"{base_dir}/sac_genetic_ibm"
        tdmpc2_path = f"{base_dir}/tdmpc2_genetic_ibm.pth"
        varmpc_path = f"{base_dir}/varmpc_genetic_ibm.pth"
        norm_path = f"{base_dir}/recurrent_vec_normalize_beta_D1stuck.pkl"

        # Setup Env
        if env_name == "heavy":
            env_cls = HeavyPhotobioreactorEnv
        elif env_name == "genetic":
            env_cls = GeneticPhotobioreactorEnv
        elif env_name == "light":
            env_cls = LightPhotobioreactorEnv
        elif env_name == "total":
            env_cls = TotalPhotobioreactorEnv
        else:
            raise ValueError(f"Unknown environment: {env_name}")
            
        print(f"Using Environment: {env_name} ({env_cls.__name__})")
        
        if env_name == "light":
            env_kwargs = {"max_cells": 300_000, "initial_cells": 3000}
        else:
            env_kwargs = {"max_cells": 300_000, "initial_cells": 3000, "difficulty": 2}

        self.env = DummyVecEnv([lambda: ActionSmoothnessWrapper(env_cls(**env_kwargs))])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=False, clip_obs=100.0)
        try:
            self.env = VecNormalize.load(norm_path, self.env.venv)
            self.env.training = False
            self.env.norm_reward = False
            print(f"✓ Normalization stats loaded from {norm_path}")
        except FileNotFoundError:
            print(f"⚠ No normalization stats found at {norm_path}")

        # Raw env for agents that don't use VecNormalize
        self.raw_env = ActionSmoothnessWrapper(env_cls(**env_kwargs))

        # Load Agents
        self.agents = {}

        # PPO (Standard -> RecurrentPPO)
        print(f"PPO Model Path configured to: {recurrent_ppo_path}")
        if enable_ppo and os.path.exists(recurrent_ppo_path + ".zip"):
            try:
                self.agents['ppo'] = RecurrentPPO.load(recurrent_ppo_path, env=self.env)
                print("✓ Loaded PPO (RecurrentPPO)")
            except Exception as e:
                print(f"✗ Skipped PPO: {e}")

        # SAC
        if enable_sac and os.path.exists(sac_path + ".zip"):
            try:
                self.agents['sac'] = SAC.load(sac_path, env=self.env)
                print("✓ Loaded SAC")
            except Exception as e:
                print(f"✗ Skipped SAC: {e}")

        # TD-MPC2
        if enable_tdmpc2 and os.path.exists(tdmpc2_path):
            try:
                agent = TDMPC2AgentClass(obs_dim=TDMPC2_OBS_DIM, action_dim=4, device=str(self.device))
                agent.load(tdmpc2_path)
                self.agents['tdmpc2'] = agent
                print("✓ Loaded TD-MPC2")
            except Exception as e:
                print(f"✗ Skipped TD-MPC2: {e}")

        # Var-MPC
        if enable_varmpc and os.path.exists(varmpc_path):
            try:
                agent = VarMPCAgentClass(obs_dim=VARMPC_OBS_DIM, action_dim=4, device=str(self.device))
                agent.load(varmpc_path)
                self.agents['varmpc'] = agent
                print("✓ Loaded Var-MPC")
            except Exception as e:
                print(f"✗ Skipped Var-MPC: {e}")

        # Random Baseline
        if enable_random:
            self.agents['random'] = RandomAgent(self.raw_env.action_space)
            print("✓ Loaded Random Baseline")


    def evaluate_episode(self, agent_name, num_steps=30000):
        """Evaluate a single episode for the given agent."""
        model = self.agents.get(agent_name)
        if not model:
            return None

        # Random and MPC agents use raw env
        if agent_name in ('random', 'tdmpc2', 'varmpc'):
            active_env = self.raw_env.unwrapped if agent_name == 'random' else self.raw_env
            raw_obs, _ = active_env.reset()
            total_reward = 0.0
            done = False
            step = 0
            strain_params = active_env.unwrapped.strain_params
            co2_actions = []  # Track CO2 actions

            # Initialize frame buffers for MPC agents
            from collections import deque
            if agent_name in ('tdmpc2', 'varmpc'):
                from TD_MPC2 import ObservationBuffer
                obs_buf = ObservationBuffer(obs_dim=TDMPC2_OBS_DIM if agent_name == 'tdmpc2' else VARMPC_OBS_DIM, order=16)
                obs_buf.reset(raw_obs, device=self.device)
                obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, m_t = model.compressor(obs_tensor, obs_buf.get_state())
                obs_buf.set_state(m_t)

            while not done and step < num_steps:
                if agent_name == 'random':
                    action, _ = model.predict(raw_obs, deterministic=True)
                elif agent_name in ('tdmpc2', 'varmpc'):
                    # Get compressed observation
                    obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    z_obs, m_t = model.compressor(obs_tensor, obs_buf.get_state())
                    obs_buf.set_state(m_t)
                    # Plan with MPPI
                    action = model.plan(z_obs.squeeze(0), horizon=24, num_samples=256, num_iters=2)
                else:
                    action, _ = model.predict(raw_obs, deterministic=True)

                # Convert CO2 action to mL/min and track
                co2_max = 120.0 if self.raw_env.unwrapped.__class__.__name__ in ['LightPhotobioreactorEnv', 'GeneticPhotobioreactorEnv'] else 5.0
                co2_ml_min = np.interp(action[3], [-1, 1], [0, co2_max])
                co2_actions.append(co2_ml_min)

                raw_obs, reward, terminated, truncated, info = active_env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

            final_pop = getattr(active_env.unwrapped, 'num_active', 0)
            avg_co2 = np.mean(co2_actions) if co2_actions else 0.0
            return {
                "Agent": agent_name,
                "Total_Reward": total_reward,
                "Final_Pop": final_pop,
                "Steps": step,
                "T_opt": strain_params.get('T_opt', float('nan')),
                "Avg_CO2_mL_min": avg_co2,
            }

        # PPO and SAC use VecNormalize
        obs = self.env.reset()
        strain_params = self.env.envs[0].unwrapped.strain_params

        total_reward = 0
        done = False
        step = 0
        lstm_states = None
        episode_start = [True]
        co2_actions = []  # Track CO2 actions

        while not done and step < num_steps:
            if agent_name == 'ppo':
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            else:  # SAC
                action, _ = model.predict(obs, deterministic=True)

            # Convert CO2 action to mL/min and track
            co2_max = 120.0 if self.env.envs[0].unwrapped.__class__.__name__ in ['LightPhotobioreactorEnv', 'GeneticPhotobioreactorEnv'] else 5.0
            co2_ml_min = np.interp(action[0][3], [-1, 1], [0, co2_max])
            co2_actions.append(co2_ml_min)

            obs, reward, dones, info = self.env.step(action)
            done = dones[0]
            total_reward += reward[0]
            step += 1
            episode_start = [done]

        if 'pop' in info[0]:
            final_pop = info[0]['pop']
        else:
            final_pop = self.env.envs[0].num_active

        avg_co2 = np.mean(co2_actions) if co2_actions else 0.0
        return {
            "Agent": agent_name,
            "Total_Reward": total_reward,
            "Final_Pop": final_pop,
            "Steps": step,
            "T_opt": strain_params.get('T_opt', float('nan')),
            "Avg_CO2_mL_min": avg_co2,
        }


    def run_benchmark(self, num_episodes=3, num_steps=30000):
        """Run benchmark across all loaded agents."""
        results = []
        print(f"\n{'='*60}")
        print(f"  BENCHMARK: {num_episodes} episodes × {num_steps:,} steps")
        print(f"{'='*60}\n")

        agent_names = list(self.agents.keys())

        for name in agent_names:
            print(f"\n[{name.upper()}]")
            for i in range(num_episodes):
                res = self.evaluate_episode(name, num_steps=num_steps)
                if res:
                    results.append(res)
                    print(f"  Episode {i+1}/{num_episodes}: "
                          f"Reward={res['Total_Reward']:.1f}, "
                          f"Pop={res['Final_Pop']:,}, "
                          f"Steps={res['Steps']:,}, "
                          f"CO2={res['Avg_CO2_mL_min']:.2f} mL/min, "
                          f"T_opt={res['T_opt']:.1f}°C")

        df = pd.DataFrame(results)
        if not df.empty:
            print(f"\n{'='*60}")
            print("  SUMMARY (Mean across episodes)")
            print(f"{'='*60}")
            summary = df.groupby("Agent")[["Total_Reward", "Final_Pop", "Steps", "Avg_CO2_mL_min"]].mean()
            summary_str = summary.to_string()
            print(summary_str)
            print(f"\n{'='*60}\n")

            # Save CSV
            csv_file = "benchmark_results.csv"
            df.to_csv(csv_file, index=False)
            print(f"CSV Results saved to {csv_file}")

            # Save TXT
            txt_file = "benchmark_results.txt"
            with open(txt_file, "w") as f:
                f.write("="*60 + "\n")
                f.write("  EVALUATION SUMMARY (Mean across episodes)\n")
                f.write("="*60 + "\n")
                f.write(summary_str + "\n")
                f.write("\n" + "="*60 + "\n")
                f.write("  DETAILED EPISODE RESULTS\n")
                f.write("="*60 + "\n")
                f.write(df.to_string(index=False) + "\n")
            print(f"Text Results saved to {txt_file}")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple RL agents on Photobioreactor Envs")
    parser.add_argument("--env", type=str, default="heavy", choices=["heavy", "genetic", "light", "total"],
                        help="Environment to evaluate on (default: heavy)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to specific PPO model (without .zip)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes per agent (default: 3)")
    parser.add_argument("--steps", type=int, default=30000,
                        help="Maximum steps per episode (default: 30000)")
    parser.add_argument("--no-ppo", action="store_true",
                        help="Disable Recurrent PPO evaluation")
    parser.add_argument("--no-sac", action="store_true",
                        help="Disable SAC evaluation")
    parser.add_argument("--no-tdmpc2", action="store_true",
                        help="Disable TD-MPC2 evaluation")
    parser.add_argument("--no-varmpc", action="store_true",
                        help="Disable Var-MPC evaluation")
    parser.add_argument("--no-random", action="store_true",
                        help="Disable random baseline evaluation")

    args = parser.parse_args()

    evaluator = Evaluator(
        env_name=args.env,
        ppo_model_path=args.model,
        enable_ppo=not args.no_ppo,
        enable_sac=not args.no_sac,
        enable_tdmpc2=not args.no_tdmpc2,
        enable_varmpc=not args.no_varmpc,
        enable_random=not args.no_random
    )
    evaluator.run_benchmark(num_episodes=args.episodes, num_steps=args.steps)

