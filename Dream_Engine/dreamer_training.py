"""
Dreamer Training Loop for GeneticPhotobioreactorEnv

This script implements the full training loop for the Dreamer World Model and Actor-Critic.
It supports both the Full and Lightweight variants of the model.

Key Components:
1. Replay Buffer: Stores experience trajectories.
2. World Model Training: Updates RSSM, Decoders, and Ensemble using collected data.
3. Behavior Training: Updates Actor/Critic using imagined trajectories in latent space.
4. Environment Interaction: Collects data using current policy (with exploration).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
from typing import Dict, List, Tuple

# Import Environment
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PPO_IBM', 'environments'))
from genetic_env import GeneticPhotobioreactorEnv

# Import Dreamer Models
from dream_model import (
    DreamerWorldModel, LightweightDreamerWorldModel,
    DreamerActor, LightweightDreamerActor,
    DreamerCritic, LightweightDreamerCritic,
    compute_lambda_returns,
    symlog,
    symexp
)

# --- CONFIGURATION ---
# Set to True for faster training with slightly reduced performance
USE_LIGHTWEIGHT = True  
# Set to True for Plan-to-Explore stochastic policies and intrinsic rewards
USE_CURIOUS_EXPLORATION = False

# Hyperparameters
BATCH_SIZE = 16          # Number of sequences per batch
SEQ_LEN = 50             # Length of sequences for world model training
HORIZON = 15  # Imagination horizon (Increased from 8 for better Biology planning)
BUFFER_SIZE = 10000      # Episodes to keep
PREFILL_STEPS = 5000     # Random steps before training
TRAIN_STEPS = 400000     # Total environment steps (Match PPO/RecurrentPPO)
TRAIN_EVERY = 5          # Train every N steps
MODEL_LR = 1e-4
ACTOR_LR = 8e-5
CRITIC_LR = 8e-5
GRAD_CLIP = 100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = f"dreamer_{'light' if USE_LIGHTWEIGHT else 'full'}_tensorboard"
MODEL_DIR = f"model_data/dreamer_{'light' if USE_LIGHTWEIGHT else 'full'}"

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Curriculum Schedule ─────────────────────────────────────────────────────
# Dreamer handles curriculum differently from PPO/TD-MPC2:
# Rather than hard-swapping environments mid-training, we train one phase per run
# and the auto-resume checkpointing carries the world model weights forward.
# Phase 2 (Hard, Full Physics) benefits most from the world model pre-built in Phase 0/1.

CURRICULUM = [
    {"difficulty": 0, "steps": 150_000, "initial_cells": 5000, "threshold": 50.0,  "pop_floor": 5000, "name": "Easy   (Baseline Physics)"},
    {"difficulty": 1, "steps": 200_000, "initial_cells": 3000, "threshold": 100.0, "pop_floor": 8000, "name": "Medium (Thermal + O2 Toxicity)"},
    {"difficulty": 2, "steps": 300_000, "initial_cells": 3000, "threshold": None,  "pop_floor": 0,    "name": "Hard   (Full Physics + Shear)"},
]


# --- REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seq_len: int):
        self.capacity = capacity
        # Store as list of episodes. Each episode is a dict of arrays.
        self.episodes = deque(maxlen=capacity)
        self.seq_len = seq_len
        self.total_steps = 0

    def add(self, episode: Dict[str, np.ndarray]):
        self.episodes.append(episode)
        self.total_steps += len(episode['reward'])

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        # Sample random episodes
        episodes = random.choices(self.episodes, k=batch_size)
        
        batch = {k: [] for k in episodes[0].keys()}
        
        for ep in episodes:
            # Sample random start index for sequence
            ep_len = len(ep['reward'])
            if ep_len <= self.seq_len:
                # Pad if too short (should be rare with 7200 step episodes)
                pad_len = self.seq_len - ep_len
                start = 0
                for k, v in ep.items():
                    # v is shaped [T, ...]
                    # We need to pad the first dimension
                    pad_shape = [(0, pad_len)] + [(0, 0)] * (v.ndim - 1)
                    padded = np.pad(v, pad_shape, mode='constant')
                    batch[k].append(padded)
            else:
                try:
                    start = np.random.randint(0, ep_len - self.seq_len)
                except ValueError:
                    start = 0 # Fallback
                
                for k, v in ep.items():
                    batch[k].append(v[start : start + self.seq_len])
                
        # Convert to tensors
        return {
            k: torch.tensor(np.stack(v), dtype=torch.float32).to(DEVICE)
            for k, v in batch.items()
        }

    def save(self, path: str):
        import pickle
        print(f"Saving ReplayBuffer to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(list(self.episodes), f)
        print("ReplayBuffer saved.")

    def load(self, path: str):
        import pickle
        if os.path.exists(path):
            print(f"Loading ReplayBuffer from {path}...")
            with open(path, 'rb') as f:
                episodes_list = pickle.load(f)
                self.episodes = deque(episodes_list, maxlen=self.capacity)
                self.total_steps = sum(len(ep['reward']) for ep in self.episodes)
            print(f"ReplayBuffer loaded with {len(self.episodes)} episodes.")
        else:
            print("No ReplayBuffer found, starting empty.")
    
    def __len__(self):
        return len(self.episodes)


# --- TRAINING WRAPPER ---
def main():
    """
    Full automated curriculum training loop.
    Runs all 3 phases in sequence: Easy (D0) → Medium (D1) → Hard (D2).
    Model weights, optimizers and replay buffer carry across phases.
    Only the environment and RSSM hidden state are reset between phases.
    """
    import json
    from tqdm import tqdm
    import sys

    print(f"─── Dreamer Curriculum Training ({'Lightweight' if USE_LIGHTWEIGHT else 'Full'}) ───")
    print(f"Device  : {DEVICE}  |  Phases: {len(CURRICULUM)}")

    obs_dim    = 7   # GeneticPhotobioreactorEnv always 7 obs
    action_dim = 4   # 4 continuous actions

    # ── Build models once — weights carry across all phases ────────────────
    if USE_LIGHTWEIGHT:
        world_model = LightweightDreamerWorldModel(obs_dim, action_dim).to(DEVICE)
        actor  = LightweightDreamerActor(feature_dim=384,  action_dim=action_dim).to(DEVICE)
        critic = LightweightDreamerCritic(feature_dim=384).to(DEVICE)
    else:
        world_model = DreamerWorldModel(obs_dim, action_dim).to(DEVICE)
        actor  = DreamerActor(feature_dim=1280,  action_dim=action_dim).to(DEVICE)
        critic = DreamerCritic(feature_dim=1280).to(DEVICE)

    world_optimizer  = optim.Adam(world_model.parameters(), lr=MODEL_LR)
    actor_optimizer  = optim.Adam(actor.parameters(),       lr=ACTOR_LR)
    critic_optimizer = optim.Adam(critic.parameters(),      lr=CRITIC_LR)

    # ── Load checkpoint ────────────────────────────────────────────────────
    checkpoint_wm     = f"{MODEL_DIR}/world_model.pth"
    checkpoint_actor  = f"{MODEL_DIR}/actor.pth"
    checkpoint_critic = f"{MODEL_DIR}/critic.pth"
    checkpoint_wm_opt = f"{MODEL_DIR}/world_opt.pth"
    checkpoint_ao_opt = f"{MODEL_DIR}/actor_opt.pth"
    checkpoint_co_opt = f"{MODEL_DIR}/critic_opt.pth"

    if os.path.exists(checkpoint_wm) and os.path.exists(checkpoint_actor):
        print(f"  Loading checkpoints from {MODEL_DIR}...")
        try:
            world_model.load_state_dict(torch.load(checkpoint_wm,     map_location=DEVICE))
            actor.load_state_dict(      torch.load(checkpoint_actor,  map_location=DEVICE))
            critic.load_state_dict(     torch.load(checkpoint_critic, map_location=DEVICE))
            if os.path.exists(checkpoint_wm_opt):
                world_optimizer.load_state_dict( torch.load(checkpoint_wm_opt, map_location=DEVICE))
                actor_optimizer.load_state_dict( torch.load(checkpoint_ao_opt, map_location=DEVICE))
                critic_optimizer.load_state_dict(torch.load(checkpoint_co_opt, map_location=DEVICE))
                print("  ✔ Weights + optimizers loaded. Resuming.")
            else:
                print("  ✔ Weights loaded.")
        except Exception as e:
            print(f"  ⚠ Checkpoint load failed: {e}. Starting from scratch.")
    else:
        print("  No checkpoints found. Starting from scratch.")

    # ── Shared replay buffer (carries across phases) ───────────────────
    buffer = ReplayBuffer(BUFFER_SIZE, obs_dim, action_dim, SEQ_LEN)
    checkpoint_buffer = f"{MODEL_DIR}/buffer.pkl"
    if os.path.exists(checkpoint_buffer):
        buffer.load(checkpoint_buffer)

    writer     = SummaryWriter(LOG_DIR)
    state_path = f"{MODEL_DIR}/training_state.json"

    # ── Load saved curriculum position ───────────────────────────────
    global_step = buffer.total_steps
    start_phase = 0
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                saved = json.load(f)
                if saved.get('global_step', 0) > global_step:
                    global_step = saved['global_step']
                start_phase = saved.get('phase_idx', 0)
                print(f"  Resuming from Phase {start_phase}, Step {global_step:,}")
        except Exception as e:
            print(f"  ⚠ Failed to load training state: {e}")

    # ═══════════════════════════════════════════════════════
    #  CURRICULUM LOOP
    # ═══════════════════════════════════════════════════════
    for phase_idx, phase in enumerate(CURRICULUM):
        if phase_idx < start_phase:
            continue   # Skip phases already completed before the crash/restart

        diff         = phase["difficulty"]
        phase_budget = phase["steps"]
        threshold    = phase["threshold"]
        pop_floor    = phase["pop_floor"]

        print(f"\n{'='*55}")
        print(f"  Phase {phase_idx} | D{diff}: {phase['name']}")
        print(f"  Budget: {phase_budget:,}  Threshold: {threshold}  Pop floor: {pop_floor}")
        print(f"{'='*55}")

        # ── Create env for this phase (new difficulty + initial cells) ───
        env = GeneticPhotobioreactorEnv(
            max_cells=300_000,
            initial_cells=phase["initial_cells"],
            difficulty=diff
        )

        # ── Reset RSSM for new physics (weights are preserved) ────────
        rssm_state  = world_model.rssm.initial_state(1, DEVICE)
        prev_action = torch.zeros(1, action_dim).to(DEVICE)

        obs, _ = env.reset()
        current_episode = {'obs': [obs], 'action': [], 'reward': [], 'continue': []}

        phase_step     = 0
        episode_idx    = 0
        recent_rewards = []
        recent_pops    = []
        WINDOW         = 4   # Rolling window (matches ~5 max eps per phase budget)
        MIN_EPS        = 4   # Must finish at least 4 eps before any advancement

        pbar = tqdm(total=phase_budget, desc=f"Phase {phase_idx} D{diff}",
                    unit="step", file=sys.stdout, mininterval=2.0)

        while phase_step < phase_budget:

            # ── Action selection ───────────────────────────────────────
            if global_step < PREFILL_STEPS:
                # Bootstrap expert heuristic during prefill (Phase 0 only)
                current_od  = obs[0]
                current_ph  = obs[1]
                current_nut = obs[2]
                k_red_estimate = 0.5 + (10.0 * current_od)
                target_light_uE = min(500.0 / np.exp(-k_red_estimate * 0.05), 2000.0)
                light_action = np.interp(target_light_uE, [0, 2000], [-1.0, 1.0])
                stir_action  = min(0.1 + (0.01 * current_od), 0.19)
                if current_nut < 800:   nut_action = 1.0
                elif current_nut < 1000: nut_action = 0.0
                else:                    nut_action = -1.0
                if current_ph > 9.5:   co2_action = 1.0
                elif current_ph > 9.0: co2_action = 0.0
                else:                  co2_action = -1.0
                action = np.array([stir_action, light_action, nut_action, co2_action], dtype=np.float32)
                action += np.random.normal(0, 0.2, size=action_dim)
                action = np.clip(action, -1.0, 1.0)
                action[0] = min(action[0], 0.19)   # Shear guard
                if co2_action == -1.0: action[3] = -1.0
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    rssm_state, _ = world_model.rssm.observe_step(rssm_state, prev_action, obs_tensor)
                    feature = world_model.rssm.get_features(rssm_state)
                    mean, std = actor(feature)
                    if USE_CURIOUS_EXPLORATION:
                        dist   = torch.distributions.Normal(mean, std)
                        action = np.tanh(dist.sample().cpu().numpy()[0])
                        noise_scale = max(0.1, 1.0 - (global_step - PREFILL_STEPS) / (sum(p['steps'] for p in CURRICULUM) * 0.5))
                        if random.random() < noise_scale:
                            action += np.random.normal(0, 0.3, size=action_dim)
                    else:
                        action = np.tanh(mean.cpu().numpy()[0])
                        if random.random() < 0.1:
                            action += np.random.normal(0, 0.1, size=action_dim)
                    action = np.clip(action, -1.0, 1.0)
                    prev_action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # ── Step environment ───────────────────────────────────────
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_episode['action'].append(action)
            current_episode['reward'].append(reward)
            current_episode['continue'].append(0.0 if done else 1.0)
            current_episode['obs'].append(next_obs)

            obs          = next_obs
            global_step  += 1
            phase_step   += 1
            pbar.update(1)

            # ── Episode end ──────────────────────────────────────────
            if done:
                current_episode['obs']      = np.array(current_episode['obs'][:-1])
                current_episode['action']   = np.array(current_episode['action'])
                current_episode['reward']   = np.array(current_episode['reward'])
                current_episode['continue'] = np.array(current_episode['continue'])

                ep_reward = float(np.sum(current_episode['reward']))
                ep_pop    = int(getattr(env, 'num_active', 0))
                recent_rewards.append(ep_reward)
                recent_pops.append(ep_pop)
                buffer.add(current_episode)
                episode_idx += 1

                print(f"  [Ph{phase_idx}|Ep{episode_idx}] R={ep_reward:.1f}  Pop={ep_pop}  "
                      f"AvgR={np.mean(recent_rewards[-WINDOW:]):.1f}  "
                      f"AvgPop={np.mean(recent_pops[-WINDOW:]):.0f}")
                writer.add_scalar(f"phase{phase_idx}/episode_reward", ep_reward, global_step)

                # Checkpoint every episode
                torch.save(world_model.state_dict(), checkpoint_wm)
                torch.save(actor.state_dict(),       checkpoint_actor)
                torch.save(critic.state_dict(),      checkpoint_critic)
                torch.save(world_optimizer.state_dict(),  checkpoint_wm_opt)
                torch.save(actor_optimizer.state_dict(),  checkpoint_ao_opt)
                torch.save(critic_optimizer.state_dict(), checkpoint_co_opt)
                buffer.save(checkpoint_buffer)
                with open(state_path, 'w') as f:
                    json.dump({'global_step': global_step, 'episode_idx': episode_idx,
                               'phase_idx': phase_idx}, f)

                # Reset for next episode
                obs, _ = env.reset()
                current_episode = {'obs': [obs], 'action': [], 'reward': [], 'continue': []}
                rssm_state  = world_model.rssm.initial_state(1, DEVICE)
                prev_action = torch.zeros(1, action_dim).to(DEVICE)

                # ── Dual-gate curriculum advancement check ───────────────
                if threshold is not None and episode_idx >= MIN_EPS and len(recent_rewards) >= WINDOW:
                    mean_rew = np.mean(recent_rewards[-WINDOW:])
                    mean_pop = np.mean(recent_pops[-WINDOW:])
                    if mean_rew >= threshold and mean_pop >= pop_floor:
                        print(f"  ✔ Phase {phase_idx} mastered! "
                              f"AvgR={mean_rew:.1f}≥{threshold} AND "
                              f"AvgPop={mean_pop:.0f}≥{pop_floor} — advancing!")
                        break
                    elif mean_rew >= threshold:
                        print(f"  ⚠ Reward ok but pop {mean_pop:.0f}<{pop_floor} — holding phase.")

            # ── World Model + Actor-Critic training ──────────────────────
            if global_step >= PREFILL_STEPS and global_step % TRAIN_EVERY == 0 and len(buffer) > 1:
                batch = buffer.sample(BATCH_SIZE)
                world_outputs = world_model.observe(
                    batch['obs'], batch['action'], batch['reward'], batch['continue']
                )
                world_optimizer.zero_grad()
                world_outputs['total_loss'].backward()
                nn.utils.clip_grad_norm_(world_model.parameters(), GRAD_CLIP)
                world_optimizer.step()

                feat          = world_outputs['features'].detach()
                flattened_feat = feat.reshape(-1, feat.shape[-1])
                h_dim = 128 if USE_LIGHTWEIGHT else 256
                h = flattened_feat[:, :h_dim]
                z = flattened_feat[:, h_dim:]
                imagined = world_model.imagine({'h': h, 'z': z}, actor,
                                               horizon=HORIZON, sample=USE_CURIOUS_EXPLORATION)

                img_feat   = imagined['features']
                img_reward = imagined['rewards']
                img_cont   = imagined['continues']

                target_values = symexp(critic(img_feat).squeeze(-1))
                target_values = torch.cat([target_values, target_values[:, -1:]], dim=1)
                returns = compute_lambda_returns(img_reward, target_values, img_cont)

                actor_loss = -returns.mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
                actor_optimizer.step()

                value_pred  = critic(img_feat.detach()).squeeze(-1)
                critic_loss = F.mse_loss(value_pred, symlog(returns.detach()))
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
                critic_optimizer.step()

                if global_step % 100 == 0:
                    writer.add_scalar(f"phase{phase_idx}/world_loss",  world_outputs['total_loss'].item(), global_step)
                    writer.add_scalar(f"phase{phase_idx}/actor_loss",  actor_loss.item(),  global_step)
                    writer.add_scalar(f"phase{phase_idx}/critic_loss", critic_loss.item(), global_step)

        pbar.close()
        env.close()
        print(f"\n  Phase {phase_idx} complete ({phase_step:,} steps, {episode_idx} episodes).")

    print("\nAll phases complete. Training Finished.")
    writer.close()



def finetune_dreamer(extra_steps: int = 500_000):
    """
    Continue Dreamer training from existing checkpoints at Difficulty 2 (Full Physics).
    Since Dreamer already auto-resumes from checkpoints, this function:
      - Loads the world model, actor, critic, AND replay buffer from MODEL_DIR
      - Extends TRAIN_STEPS by extra_steps so the training loop runs longer
      - Reduces LRs to 1e-4 / 6e-5 to prevent catastrophic forgetting
      - Increases HORIZON from 15 → 25 to better capture 6-day fouling dynamics

    Note: Dreamer's own checkpoint logic already handles interruption/resumption.
    This function is mainly useful for deliberately continuing at reduced LR
    after a previously converged run.
    """
    import json
    state_path = f"{MODEL_DIR}/training_state.json"
    current_step = 0

    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                current_step = state.get('global_step', 0)
        except Exception:
            pass

    if not os.path.exists(f"{MODEL_DIR}/world_model.pth"):
        print(f"  [ERROR] No saved Dreamer model found in {MODEL_DIR}")
        print("  Run 'python dreamer_training.py' first.")
        return

    print("\u2500── Dreamer Fine-Tune (Difficulty 2, Full Physics) ───")
    print(f"  Current step   : {current_step:,}")
    print(f"  Extra steps    : {extra_steps:,}")
    print(f"  New total      : {current_step + extra_steps:,}")
    print(f"  LR change      : World 1e-4→1e-4  Actor/Critic 8e-5→6e-5")
    print(f"  Horizon change : {HORIZON} → 25 (captures slow 6-day fouling)")

    # Patch global hypers for the fine-tune run
    # We temporarily override module-level consts by passing them as args to main()
    # rather than editing globals (safer for resumable runs)
    original_train_steps = TRAIN_STEPS
    original_horizon = HORIZON

    # Write an extended training_state so main() sees the new target
    extended_state = {'global_step': current_step, 'episode_idx': 0, 'finetune_extra': extra_steps}
    with open(f"{MODEL_DIR}/finetune_state.json", 'w') as f:
        json.dump(extended_state, f)

    # Override TRAIN_STEPS so the while loop runs for extra_steps more
    import dreamer_training as _self
    _self.TRAIN_STEPS = current_step + extra_steps
    _self.HORIZON = 25            # Longer imagination for slow fouling dynamics
    _self.MODEL_LR = 1e-4         # Same as before (already conservative)
    _self.ACTOR_LR = 6e-5         # Slightly lower
    _self.CRITIC_LR = 6e-5
    print("  ✔ Hyperparameters patched. Starting fine-tune pass...")
    main()  # Auto-curriculum will see phase_idx in training_state.json and resume from Phase 2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dreamer Training for GeneticPBR")
    parser.add_argument("--finetune", action="store_true",
                        help="Load saved model and fine-tune for extra steps at Difficulty 2")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Extra steps for --finetune mode (default: 500000)")
    args = parser.parse_args()

    if args.finetune:
        finetune_dreamer(extra_steps=args.steps)
    else:
        main()  # Auto-runs all 3 curriculum phases in sequence

