"""TD3_lru.py — v48 variant of TD3.py: diagonal-LRU recurrent core + CPU-efficiency patches.
(full rationale: docs/decision_history.md#--legacy-TD3_lru-py-1)"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
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
# CPU-efficiency patches. soft_update is exactly equivalent to TD3.py's; td3_update differs
# only in fusing the policy and BC actor passes (float-level differences, ~1e-7).

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
    forward. Rows are independent given a zero initial state."""
    batch = base.sample_mixed_batch(demo_buffer, online_buffer, base.BATCH_SIZE, base.DEMO_FRACTION)
    if batch is None:
        return None, None

    critic_loss = base.critic_update(actor_target, critic, critic_target, critic_opt, batch)

    actor_loss = None
    if update_idx % base.POLICY_DELAY == 0:
        obs = batch["obs"]
        bc_obs, bc_act, _, _, _ = demo_buffer._sample_raw(base.BATCH_SIZE)
        if bc_obs:
            bc_obs_t = torch.tensor(np.array(bc_obs), dtype=torch.float32, device=DEVICE)
            bc_act_t = torch.tensor(np.array(bc_act), dtype=torch.float32, device=DEVICE)
            fused, _ = actor(torch.cat([obs, bc_obs_t], dim=0))
            sat_term = base.preact_penalty(actor.last_preact)
            pred_action, bc_pred = fused[:obs.shape[0]], fused[obs.shape[0]:]
            bc_term = base.BC_COEF * F.mse_loss(bc_pred, bc_act_t)
        else:
            pred_action, _ = actor(obs)
            sat_term = base.preact_penalty(actor.last_preact)
            bc_term = torch.tensor(0.0, device=DEVICE)

        q_pred = critic.q1_only(obs, pred_action)
        actor_loss = base.actor_step(actor, actor_target, critic, critic_target, actor_opt,
                                     -base.actor_q_weight(q_pred) * q_pred.mean() + bc_term + sat_term)

    return critic_loss, actor_loss


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
