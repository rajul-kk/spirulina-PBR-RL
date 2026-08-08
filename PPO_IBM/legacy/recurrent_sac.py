"""
Recurrent SAC (Soft Actor-Critic with LSTM) for GeneticPhotobioreactorEnv
=========================================================================

Architecture:
  - RecurrentActor   : LSTM(256) → GaussianPolicy (re-parameterised)
  - RecurrentCritic  : Twin independent LSTM(256) soft Q-networks
  - SequenceBuffer   : Episode-based replay; samples fixed-length sequences
  - Alpha            : Automatic entropy coefficient (learned online)
  - Curriculum       : 3-phase loop (D0 → D1 → D2) matching PPO / TD-MPC2

Why Recurrent SAC over vanilla SAC?
  The photobioreactor is a POMDP — single observations don't fully reveal the
  state (biofouling accumulates invisibly, O2 lags, pH has inertia).
  LSTM maintains a hidden belief state h_t across the full episode, letting
  the policy reason about trends rather than just snapshots.

Usage:
  python recurrent_sac.py                  # full curriculum
  python recurrent_sac.py --finetune       # continue from checkpoint at D2
  python recurrent_sac.py --finetune --steps 1000000
"""

import os, sys, json, random, argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ── Environment ──────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "environments"))
from genetic_env import GeneticPhotobioreactorEnv

# ── Hyper-parameters ─────────────────────────────────────────────────────────
HIDDEN_DIM   = 128          # LSTM hidden size
LSTM_LAYERS  = 1            # LSTM depth (1 is standard; 2 adds capacity)
BATCH_SIZE   = 32           # Sequences sampled per update
SEQ_LEN      = 50           # BPTT sequence length (steps)
BUFFER_SIZE  = 5_000        # Max episodes to keep
LR_ACTOR     = 3e-4
LR_CRITIC    = 3e-4
LR_ALPHA     = 3e-4
GAMMA        = 0.99
TAU          = 0.005        # Polyak update coefficient
GRAD_CLIP    = 5.0
TRAIN_EVERY  = 1            # Update every N environment steps
UPDATES_PER_STEP = 1        # Gradient updates per environment step
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "model_data/recurrent_sac"
LOG_DIR   = "rsac_tensorboard"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Curriculum (mirrors recurrent_ppo.py / TD_MPC2.py) ──────────────────────
CURRICULUM = [
    {"difficulty": 0, "steps":  60_000, "initial_cells": 5000,
     "threshold": 50.0,  "pop_floor": 5000, "name": "Easy   (Baseline)"},
    {"difficulty": 1, "steps":  80_000, "initial_cells": 3000,
     "threshold": 100.0, "pop_floor": 8000, "name": "Medium (O2 Toxicity)"},
    {"difficulty": 2, "steps": 100_000, "initial_cells": 3000,
     "threshold": None,  "pop_floor": 0,    "name": "Hard   (Full Physics)"},
]
MIN_EPS_BEFORE_ADVANCE = 8   # Must complete this many eps before dual-gate check
WINDOW = 8                   # Rolling window for mean reward / pop

# ═════════════════════════════════════════════════════════════════════════════
#  NETWORKS
# ═════════════════════════════════════════════════════════════════════════════

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class RecurrentActor(nn.Module):
    """
    LSTM-based Gaussian policy that maps observation sequence → (mean, log_std).
    Re-parameterised sampling + tanh squashing for bounded action space [-1, 1].
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: int = HIDDEN_DIM, lstm_layers: int = LSTM_LAYERS):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.lstm_layers = lstm_layers

        # Input projection (normalises raw obs before LSTM)
        self.input_fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True)

        # Gaussian head
        self.mean_fc    = nn.Linear(hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(hidden_dim, action_dim)

    def initial_hidden(self, batch: int) -> tuple:
        h = torch.zeros(self.lstm_layers, batch, self.hidden_dim).to(DEVICE)
        c = torch.zeros(self.lstm_layers, batch, self.hidden_dim).to(DEVICE)
        return (h, c)

    def forward(self, obs: torch.Tensor, hidden: tuple | None = None):
        """
        Args:
            obs    : [B, T, obs_dim]  or  [1, 1, obs_dim] during inference
            hidden : (h, c) LSTM state; None → zero-init

        Returns:
            mean, log_std  : [B, T, action_dim]
            hidden         : updated (h, c) for next step
        """
        B, T, _ = obs.shape
        if hidden is None:
            hidden = self.initial_hidden(B)

        x   = self.input_fc(obs.reshape(B * T, -1)).reshape(B, T, -1)
        out, hidden = self.lstm(x, hidden)

        mean    = self.mean_fc(out)
        log_std = self.log_std_fc(out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, hidden

    def sample(self, obs: torch.Tensor, hidden: tuple | None = None):
        """
        Returns:
            action    : tanh-squashed sample  [B, T, action_dim]
            log_prob  : log π(a|s) corrected for tanh  [B, T, 1]
            mean      : deterministic action (for eval)
            hidden    : updated LSTM state
        """
        mean, log_std, hidden = self.forward(obs, hidden)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t  = dist.rsample()                               # re-param trick
        y_t  = torch.tanh(x_t)

        # Tanh-correction to log-prob (Appendix C, Haarnoja et al. 2018)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)           # [B, T, 1]

        mean_action = torch.tanh(mean)
        return y_t, log_prob, mean_action, hidden


class RecurrentCritic(nn.Module):
    """
    Twin soft Q-networks with independent LSTM encoders.
    Each network maps (obs, action) sequence → Q-value sequence.
    Twin architecture prevents overestimation bias (Fujimoto et al. 2018).
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: int = HIDDEN_DIM, lstm_layers: int = LSTM_LAYERS):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.lstm_layers = lstm_layers
        inp = obs_dim + action_dim

        # Q1
        self.q1_input = nn.Sequential(nn.Linear(inp, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.q1_lstm  = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True)
        self.q1_head  = nn.Linear(hidden_dim, 1)

        # Q2
        self.q2_input = nn.Sequential(nn.Linear(inp, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.q2_lstm  = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True)
        self.q2_head  = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch: int) -> tuple:
        zeros = lambda: torch.zeros(self.lstm_layers, batch, self.hidden_dim).to(DEVICE)
        return (zeros(), zeros()), (zeros(), zeros())

    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                hidden1: tuple | None = None, hidden2: tuple | None = None):
        """
        Args:
            obs, action : [B, T, dim]
        Returns:
            q1, q2      : [B, T, 1]
            hidden1, hidden2 : updated LSTM states
        """
        B, T, _ = obs.shape
        if hidden1 is None:
            hidden1, hidden2 = self.initial_hidden(B)

        sa = torch.cat([obs, action], dim=-1)               # [B, T, obs+act]
        sa_flat = sa.reshape(B * T, -1)

        x1 = self.q1_input(sa_flat).reshape(B, T, -1)
        x2 = self.q2_input(sa_flat).reshape(B, T, -1)

        out1, hidden1 = self.q1_lstm(x1, hidden1)
        out2, hidden2 = self.q2_lstm(x2, hidden2)

        q1 = self.q1_head(out1)                             # [B, T, 1]
        q2 = self.q2_head(out2)

        return q1, q2, hidden1, hidden2

    def q1(self, obs, action, hidden=None):
        B, T, _ = obs.shape
        if hidden is None:
            hidden = (torch.zeros(self.lstm_layers, B, self.hidden_dim).to(DEVICE),
                      torch.zeros(self.lstm_layers, B, self.hidden_dim).to(DEVICE))
        sa  = torch.cat([obs, action], dim=-1).reshape(B * T, -1)
        x   = self.q1_input(sa).reshape(B, T, -1)
        out, hidden = self.q1_lstm(x, hidden)
        return self.q1_head(out), hidden


# ═════════════════════════════════════════════════════════════════════════════
#  SEQUENCE REPLAY BUFFER
# ═════════════════════════════════════════════════════════════════════════════

class SequenceReplayBuffer:
    """
    Stores complete episodes. Samples random fixed-length sub-sequences for
    training. BPTT is truncated to SEQ_LEN steps; hidden states are
    zero-initialised at the start of each sampled sequence.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seq_len: int):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.seq_len    = seq_len
        self.episodes   = deque(maxlen=capacity)
        self.total_steps = 0

    def add(self, episode: dict):
        """episode: dict with keys obs, action, reward, next_obs, done (np arrays)"""
        self.episodes.append(episode)
        self.total_steps += len(episode["reward"])

    def sample(self, batch_size: int) -> dict:
        """Returns a dict of tensors with shape [batch_size, seq_len, dim]"""
        obs_batch      = []
        act_batch      = []
        rew_batch      = []
        next_obs_batch = []
        done_batch     = []

        while len(obs_batch) < batch_size:
            ep = random.choice(self.episodes)
            ep_len = len(ep["reward"])
            if ep_len < self.seq_len:
                continue  # Skip episodes shorter than one sequence
            start = random.randint(0, ep_len - self.seq_len)
            sl    = slice(start, start + self.seq_len)
            obs_batch.append(ep["obs"][sl])
            act_batch.append(ep["action"][sl])
            rew_batch.append(ep["reward"][sl])
            next_obs_batch.append(ep["next_obs"][sl])
            done_batch.append(ep["done"][sl])

        def to_tensor(lst, dtype=torch.float32):
            return torch.tensor(np.array(lst), dtype=dtype).to(DEVICE)

        return {
            "obs":      to_tensor(obs_batch),           # [B, T, obs_dim]
            "action":   to_tensor(act_batch),           # [B, T, act_dim]
            "reward":   to_tensor(rew_batch).unsqueeze(-1),  # [B, T, 1]
            "next_obs": to_tensor(next_obs_batch),      # [B, T, obs_dim]
            "done":     to_tensor(done_batch).unsqueeze(-1),  # [B, T, 1]
        }

    def __len__(self):
        return len(self.episodes)

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(list(self.episodes), f)
        print(f"  Buffer saved ({len(self.episodes)} episodes, {self.total_steps} steps).")

    def load(self, path: str):
        import pickle
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            eps = pickle.load(f)
        self.episodes   = deque(eps, maxlen=self.capacity)
        self.total_steps = sum(len(e["reward"]) for e in self.episodes)
        print(f"  Buffer loaded ({len(self.episodes)} episodes, {self.total_steps} steps).")


# ═════════════════════════════════════════════════════════════════════════════
#  SAC UPDATE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def soft_update(net: nn.Module, target_net: nn.Module, tau: float = TAU):
    for p, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def save_checkpoint(actor, critic, critic_target, actor_opt, critic_opt,
                    log_alpha, alpha_opt, buffer, state: dict):
    torch.save(actor.state_dict(),          f"{MODEL_DIR}/actor.pth")
    torch.save(critic.state_dict(),         f"{MODEL_DIR}/critic.pth")
    torch.save(critic_target.state_dict(),  f"{MODEL_DIR}/critic_target.pth")
    torch.save(actor_opt.state_dict(),      f"{MODEL_DIR}/actor_opt.pth")
    torch.save(critic_opt.state_dict(),     f"{MODEL_DIR}/critic_opt.pth")
    torch.save({"log_alpha": log_alpha.detach().cpu(),
                "alpha_opt": alpha_opt.state_dict()},
               f"{MODEL_DIR}/alpha.pth")
    buffer.save(f"{MODEL_DIR}/buffer.pkl")
    with open(f"{MODEL_DIR}/training_state.json", "w") as f:
        json.dump(state, f)


def load_checkpoint(actor, critic, critic_target, actor_opt, critic_opt,
                    log_alpha, alpha_opt, buffer, lr_actor: float, lr_critic: float):
    state = {}
    wm = f"{MODEL_DIR}/actor.pth"
    if not os.path.exists(wm):
        print("  No checkpoint found. Starting from scratch.")
        return state

    try:
        actor.load_state_dict(        torch.load(f"{MODEL_DIR}/actor.pth",         map_location=DEVICE))
        critic.load_state_dict(       torch.load(f"{MODEL_DIR}/critic.pth",        map_location=DEVICE))
        critic_target.load_state_dict(torch.load(f"{MODEL_DIR}/critic_target.pth", map_location=DEVICE))
        actor_opt.load_state_dict(    torch.load(f"{MODEL_DIR}/actor_opt.pth",     map_location=DEVICE))
        critic_opt.load_state_dict(   torch.load(f"{MODEL_DIR}/critic_opt.pth",    map_location=DEVICE))

        alpha_data = torch.load(f"{MODEL_DIR}/alpha.pth", map_location=DEVICE)
        log_alpha.data.copy_(alpha_data["log_alpha"].to(DEVICE))
        alpha_opt.load_state_dict(alpha_data["alpha_opt"])

        buffer.load(f"{MODEL_DIR}/buffer.pkl")

        sf = f"{MODEL_DIR}/training_state.json"
        if os.path.exists(sf):
            with open(sf) as f:
                state = json.load(f)
        print("  ✔ Checkpoint loaded. Resuming training.")
    except Exception as e:
        print(f"  ⚠ Checkpoint load failed: {e}. Starting fresh.")
        state = {}
    return state


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train():
    """
    Full automated curriculum: Easy → Medium → Hard.
    Model weights, replay buffer, and α carry across phases.
    Only the environment is swapped; LSTM hidden state is reset per episode.
    """
    from tqdm import tqdm

    obs_dim    = 7   # GeneticPhotobioreactorEnv
    action_dim = 3

    print(f"{'='*55}")
    print(f"  Recurrent SAC  |  Device: {DEVICE}")
    print(f"  LSTM hidden={HIDDEN_DIM} layers={LSTM_LAYERS}  seq_len={SEQ_LEN}")
    print(f"{'='*55}")

    # ── Build networks ────────────────────────────────────────────────────────
    actor         = RecurrentActor(obs_dim, action_dim).to(DEVICE)
    critic        = RecurrentCritic(obs_dim, action_dim).to(DEVICE)
    critic_target = RecurrentCritic(obs_dim, action_dim).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False                         # Target is never directly trained

    actor_opt  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # ── Automatic entropy coefficient α ──────────────────────────────────────
    target_entropy = float(-action_dim)                 # h* = -dim(A)
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha_opt = optim.Adam([log_alpha], lr=LR_ALPHA)

    # ── Shared replay buffer ──────────────────────────────────────────────────
    buffer = SequenceReplayBuffer(BUFFER_SIZE, obs_dim, action_dim, SEQ_LEN)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    saved_state = load_checkpoint(actor, critic, critic_target,
                                  actor_opt, critic_opt,
                                  log_alpha, alpha_opt, buffer,
                                  LR_ACTOR, LR_CRITIC)

    global_step = saved_state.get("global_step", 0)
    start_phase = saved_state.get("phase_idx",   0)

    writer = SummaryWriter(LOG_DIR)

    # ══════════════════════════════════════════════════════════════════════════
    #  CURRICULUM LOOP
    # ══════════════════════════════════════════════════════════════════════════
    for phase_idx, phase in enumerate(CURRICULUM):
        if phase_idx < start_phase:
            continue

        diff         = phase["difficulty"]
        phase_budget = phase["steps"]
        threshold    = phase["threshold"]
        pop_floor    = phase["pop_floor"]

        print(f"\n{'='*55}")
        print(f"  Phase {phase_idx} | D{diff}: {phase['name']}")
        print(f"  Budget: {phase_budget:,}  Reward threshold: {threshold}"
              f"  Pop floor: {pop_floor}")
        print(f"{'='*55}")

        env = GeneticPhotobioreactorEnv(
            max_cells=300_000, initial_cells=phase["initial_cells"], difficulty=diff
        )

        obs, _ = env.reset()
        # Per-episode LSTM hidden state (actor only — critics don't need inference state)
        actor_hidden = actor.initial_hidden(batch=1)

        # Current episode buffers
        ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []

        # If we are resuming mid-phase, deduct the steps we've already done
        phase_step     = 0
        episode_idx    = 0
        if phase_idx == start_phase and "phase_step" in saved_state:
            phase_step = saved_state["phase_step"]
            episode_idx = saved_state["episode_idx"]

        recent_rewards = []
        recent_pops    = []

        pbar = tqdm(total=phase_budget, initial=phase_step, desc=f"Ph{phase_idx}|D{diff}",
                    unit="step", file=sys.stdout, mininterval=2.0)

        while phase_step < phase_budget:

            # ── Action selection ──────────────────────────────────────────────
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32
                                     ).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,obs_dim]
                action_t, _, _, actor_hidden = actor.sample(obs_t, actor_hidden)
                action = action_t.squeeze().cpu().numpy()                     # [action_dim]

            # ── Environment step ──────────────────────────────────────────────
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_obs.append(obs)
            ep_act.append(action)
            ep_rew.append(reward)
            ep_nobs.append(next_obs)
            ep_done.append(float(done))

            obs          = next_obs
            global_step  += 1
            phase_step   += 1
            pbar.update(1)

            # ── Episode end ───────────────────────────────────────────────────
            if done:
                if len(ep_obs) >= SEQ_LEN:         # Only keep episodes long enough to sample
                    episode = {
                        "obs":      np.array(ep_obs,  dtype=np.float32),
                        "action":   np.array(ep_act,  dtype=np.float32),
                        "reward":   np.array(ep_rew,  dtype=np.float32),
                        "next_obs": np.array(ep_nobs, dtype=np.float32),
                        "done":     np.array(ep_done, dtype=np.float32),
                    }
                    buffer.add(episode)

                ep_reward = float(np.sum(ep_rew))
                ep_pop    = int(getattr(env, "num_active", 0))
                recent_rewards.append(ep_reward)
                recent_pops.append(ep_pop)
                episode_idx += 1

                mean_rew = np.mean(recent_rewards[-WINDOW:])
                mean_pop = np.mean(recent_pops[-WINDOW:])
                alpha_val = log_alpha.exp().item()
                print(f"\n  [Ph{phase_idx}|Ep{episode_idx}] "
                      f"R={ep_reward:.1f}  Pop={ep_pop}  "
                      f"AvgR={mean_rew:.1f}  AvgPop={mean_pop:.0f}  α={alpha_val:.4f}")
                writer.add_scalar(f"phase{phase_idx}/episode_reward", ep_reward, global_step)
                writer.add_scalar(f"phase{phase_idx}/episode_pop",    ep_pop,    global_step)

                # Checkpoint every episode
                save_checkpoint(actor, critic, critic_target,
                                actor_opt, critic_opt, log_alpha, alpha_opt, buffer,
                                {"global_step": global_step, "phase_idx": phase_idx,
                                 "episode_idx": episode_idx, "phase_step": phase_step})

                # Reset
                obs, _ = env.reset()
                actor_hidden = actor.initial_hidden(batch=1)
                ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []

                # ── Dual-gate advancement ─────────────────────────────────────
                if (threshold is not None
                        and episode_idx >= MIN_EPS_BEFORE_ADVANCE
                        and len(recent_rewards) >= WINDOW):
                    if mean_rew >= threshold and mean_pop >= pop_floor:
                        print(f"  ✔ Phase {phase_idx} mastered! "
                              f"AvgR={mean_rew:.1f}≥{threshold} AND "
                              f"AvgPop={mean_pop:.0f}≥{pop_floor} — advancing!")
                        break
                    elif mean_rew >= threshold:
                        print(f"  ⚠ Reward ok but pop {mean_pop:.0f}<{pop_floor} — holding.")

            # ── SAC Updates ───────────────────────────────────────────────────
            if (len(buffer) > 0 and global_step % TRAIN_EVERY == 0):
                for _ in range(UPDATES_PER_STEP):
                    _sac_update(actor, critic, critic_target, critic_target,
                                actor_opt, critic_opt, log_alpha, alpha_opt,
                                buffer, target_entropy, writer, global_step, phase_idx)

        pbar.close()
        env.close()
        print(f"\n  Phase {phase_idx} done ({phase_step:,} steps, {episode_idx} episodes).")

    print("\nAll curriculum phases complete!")
    writer.close()


def _sac_update(actor, critic, critic_target, _unused,
                actor_opt, critic_opt, log_alpha, alpha_opt,
                buffer, target_entropy, writer, step, phase_idx):
    """One SAC gradient update step."""
    batch = buffer.sample(BATCH_SIZE)

    obs      = batch["obs"]       # [B, T, obs_dim]
    actions  = batch["action"]    # [B, T, act_dim]
    rewards  = batch["reward"]    # [B, T, 1]
    next_obs = batch["next_obs"]  # [B, T, obs_dim]
    dones    = batch["done"]      # [B, T, 1]

    alpha = log_alpha.exp().detach()

    # ── Critic Target ─────────────────────────────────────────────────────────
    with torch.no_grad():
        next_act, next_log_pi, _, _ = actor.sample(next_obs)
        q1_next, q2_next, _, _ = critic_target(next_obs, next_act)
        min_q_next  = torch.min(q1_next, q2_next) - alpha * next_log_pi
        q_target    = rewards + GAMMA * (1.0 - dones) * min_q_next   # [B, T, 1]

    # ── Critic Loss ───────────────────────────────────────────────────────────
    q1, q2, _, _ = critic(obs, actions)
    critic_loss  = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

    critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
    critic_opt.step()

    # ── Actor Loss ────────────────────────────────────────────────────────────
    new_act, log_pi, _, _ = actor.sample(obs)
    q1_pi, _              = critic.q1(obs, new_act)
    actor_loss = (alpha * log_pi - q1_pi).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
    actor_opt.step()

    # ── Alpha (Entropy Coefficient) Loss ──────────────────────────────────────
    alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()

    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    # ── Polyak Update ─────────────────────────────────────────────────────────
    soft_update(critic, critic_target)

    # ── Logging ───────────────────────────────────────────────────────────────
    if step % 200 == 0:
        writer.add_scalar(f"phase{phase_idx}/critic_loss", critic_loss.item(), step)
        writer.add_scalar(f"phase{phase_idx}/actor_loss",  actor_loss.item(),  step)
        writer.add_scalar(f"phase{phase_idx}/alpha",       log_alpha.exp().item(), step)
        writer.add_scalar(f"phase{phase_idx}/entropy",     -log_pi.mean().item(), step)


# ═════════════════════════════════════════════════════════════════════════════
#  FINE-TUNE MODE
# ═════════════════════════════════════════════════════════════════════════════

def finetune(extra_steps: int = 500_000):
    """
    Load existing checkpoint and continue training at Difficulty 2 (Full Physics)
    with reduced learning rates to consolidate without catastrophic forgetting.

    Strategy:
      - Load actor, critic, buffer from MODEL_DIR
      - Set LR_ACTOR = LR_CRITIC = LR_ALPHA = 3e-5 (10× lower than default)
      - Run extra_steps on D2 with the same dual-gate check (no threshold)
    """
    state_path = f"{MODEL_DIR}/training_state.json"
    current_step = 0
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
            current_step = state.get("global_step", 0)

    if not os.path.exists(f"{MODEL_DIR}/actor.pth"):
        print("  [ERROR] No checkpoint found. Run 'python recurrent_sac.py' first.")
        return

    print(f"\n{'─'*55}")
    print(f"  Recurrent SAC Fine-Tune  |  D2 (Full Physics)")
    print(f"  Current step : {current_step:,}   Extra: {extra_steps:,}")
    print(f"  LR 3e-4 → 3e-5  (10× lower to prevent catastrophic forgetting)")
    print(f"{'─'*55}")

    ft_lr  = 3e-5
    obs_dim, action_dim = 7, 3

    actor         = RecurrentActor(obs_dim, action_dim).to(DEVICE)
    critic        = RecurrentCritic(obs_dim, action_dim).to(DEVICE)
    critic_target = RecurrentCritic(obs_dim, action_dim).to(DEVICE)
    for p in critic_target.parameters():
        p.requires_grad = False

    actor_opt  = optim.Adam(actor.parameters(),  lr=ft_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=ft_lr)
    log_alpha  = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha_opt  = optim.Adam([log_alpha], lr=ft_lr)

    buffer = SequenceReplayBuffer(BUFFER_SIZE, obs_dim, action_dim, SEQ_LEN)
    load_checkpoint(actor, critic, critic_target,
                    actor_opt, critic_opt, log_alpha, alpha_opt, buffer,
                    ft_lr, ft_lr)

    critic_target.load_state_dict(critic.state_dict())

    # Write a patched training_state so train() knows to start at Phase 2
    with open(state_path, "w") as f:
        json.dump({"global_step": current_step, "phase_idx": 2, "episode_idx": 0}, f)

    # Temporarily override the phase budget so Phase 2 runs for extra_steps
    CURRICULUM[2]["steps"] = extra_steps

    # Reinstate the module-level optimiser state
    import recurrent_sac as _self
    _self.LR_ACTOR  = ft_lr
    _self.LR_CRITIC = ft_lr
    _self.LR_ALPHA  = ft_lr
    print("  ✔ LRs patched. Starting fine-tune loop on D2...")
    train()


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recurrent SAC for GeneticPBR")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune existing checkpoint on D2 at lower LR")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Extra steps for --finetune mode (default: 500000)")
    args = parser.parse_args()

    if args.finetune:
        finetune(extra_steps=args.steps)
    else:
        train()
