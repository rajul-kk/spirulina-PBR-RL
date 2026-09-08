"""TD3_lru.py — v48 variant of TD3.py: diagonal-LRU recurrent core + CPU-efficiency patches.
(full rationale: docs/decision_history.md#--legacy-TD3_lru-py-1)"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thread count must be set before the first heavy op. Torch defaults to one thread per
# core, which oversubscribes badly on this loop's small (24,60,128) tensors.
if not torch.cuda.is_available():
    torch.set_num_threads(int(os.environ.get("TD3_THREADS", "6")))

import TD3 as base
from lru_core import LRUCore

DEVICE = base.DEVICE


# ═════════════════════════════════════════════════════════════════════════════
# Networks: identical to TD3.py's except the recurrent core.

class LRUActor(base.RecurrentActor):
    def __init__(self, obs_dim, action_dim, hidden_dim=base.HIDDEN_DIM, lstm_layers=base.LSTM_LAYERS):
        super().__init__(obs_dim, action_dim, hidden_dim, lstm_layers)
        self.lstm = LRUCore(hidden_dim)   # named 'lstm' so base.forward() needs no change

    def initial_hidden(self, batch):
        return self.lstm.initial_hidden(batch, device=DEVICE)


class LRUCritic(base.RecurrentCritic):
    def __init__(self, obs_dim, action_dim, hidden_dim=base.HIDDEN_DIM, lstm_layers=base.LSTM_LAYERS):
        super().__init__(obs_dim, action_dim, hidden_dim, lstm_layers)
        self.q1_lstm = LRUCore(hidden_dim)
        self.q2_lstm = LRUCore(hidden_dim)

    def initial_hidden(self, batch):
        return (self.q1_lstm.initial_hidden(batch, device=DEVICE),
                self.q2_lstm.initial_hidden(batch, device=DEVICE))

    def q1_only(self, obs, action):
        # base.q1_only hard-codes the LSTM (h, c) tuple, so it must be overridden.
        B, T, _ = obs.shape
        sa = torch.cat([obs, action], dim=-1).reshape(B * T, -1)
        x = self.q1_input(sa).reshape(B, T, -1)
        out, _ = self.q1_lstm(x, self.q1_lstm.initial_hidden(B, device=DEVICE))
        return self.q1_head(out)


# ═════════════════════════════════════════════════════════════════════════════
# CPU-efficiency patches. Both are exactly equivalent to TD3.py's versions.

@torch.no_grad()
def soft_update(net, target_net, tau=base.TAU):
    """Fused foreach Polyak averaging; replaces a Python loop over ~14 parameter tensors."""
    p = list(net.parameters())
    tp = list(target_net.parameters())
    torch._foreach_mul_(tp, 1.0 - tau)
    torch._foreach_add_(tp, p, alpha=tau)


def td3_update(actor, actor_target, critic, critic_target, actor_opt, critic_opt,
               demo_buffer, online_buffer, update_idx):
    """TD3.td3_update with the policy-batch and BC-batch actor passes fused into one
    forward. Rows are independent given a zero initial state, so this is exact."""
    batch = base.sample_mixed_batch(demo_buffer, online_buffer, base.BATCH_SIZE, base.DEMO_FRACTION)
    if batch is None:
        return None, None

    obs, actions, rewards = batch["obs"], batch["action"], batch["reward"]
    next_obs, dones = batch["next_obs"], batch["done"]

    with torch.no_grad():
        next_action, _ = actor_target(next_obs)
        noise = (torch.randn_like(next_action) * base.POLICY_NOISE).clamp(-base.NOISE_CLIP, base.NOISE_CLIP)
        next_action = (next_action + noise).clamp(-1.0, 1.0)
        q1_next, q2_next, _, _ = critic_target(next_obs, next_action)
        q_target = rewards + base.GAMMA * (1.0 - dones) * torch.min(q1_next, q2_next)

    q1, q2, _, _ = critic(obs, actions)
    critic_loss = F.huber_loss(q1, q_target, delta=1.0) + F.huber_loss(q2, q_target, delta=1.0)
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), base.GRAD_CLIP)
    critic_opt.step()

    actor_loss = None
    if update_idx % base.POLICY_DELAY == 0:
        bc_obs, bc_act, _, _, _ = demo_buffer._sample_raw(base.BATCH_SIZE)
        if bc_obs:
            bc_obs_t = torch.tensor(np.array(bc_obs), dtype=torch.float32, device=DEVICE)
            bc_act_t = torch.tensor(np.array(bc_act), dtype=torch.float32, device=DEVICE)
            fused, _ = actor(torch.cat([obs, bc_obs_t], dim=0))
            pred_action, bc_pred = fused[:obs.shape[0]], fused[obs.shape[0]:]
            bc_term = base.BC_COEF * F.mse_loss(bc_pred, bc_act_t)
        else:
            pred_action, _ = actor(obs)
            bc_term = torch.tensor(0.0, device=DEVICE)

        q_pred = critic.q1_only(obs, pred_action)
        lam = torch.clamp(base.TD3BC_ALPHA / (q_pred.abs().mean().detach() + 1e-3), max=100.0)
        actor_loss = -lam * q_pred.mean() + bc_term

        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), base.GRAD_CLIP)
        actor_opt.step()

        soft_update(actor, actor_target)
        soft_update(critic, critic_target)
        actor_loss = float(actor_loss.item())

    return float(critic_loss.item()), actor_loss


# ═════════════════════════════════════════════════════════════════════════════
# Patch TD3's module globals, then reuse its train loop verbatim. Checkpoint paths are
# redirected so an LRU run can never overwrite the LSTM baseline's artifacts.

def install():
    base.RecurrentActor = LRUActor
    base.RecurrentCritic = LRUCritic
    base.soft_update = soft_update
    base.td3_update = td3_update
    base.CHECKPOINT_DIR = "model_data/td3_lru_checkpoints"
    base.STATE_PATH = "model_data/td3_lru_training_state.pkl"
    base.BUFFER_PATH = "model_data/td3_lru_checkpoints/online_buffer.pkl"
    base.BEST_CHECKPOINT_DIR = "model_data/td3_lru_checkpoints_best"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 + diagonal LRU core")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    install()
    print(f"--- TD3+LRU | core=DiagonalLRU | threads={torch.get_num_threads()} ---")
    base.train(resume=args.resume)
