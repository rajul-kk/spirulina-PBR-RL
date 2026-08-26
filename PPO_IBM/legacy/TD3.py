"""
TD3 (Twin Delayed DDPG) for GeneticPhotobioreactorEnv.

Tests whether the "BC beats RL" pattern (experiments/bc_scaffold/) is specific to
on-policy/planning methods (PPO, TD-MPC2) or generalizes to off-policy actor-critic too.
Reuses proven project infrastructure rather than building from scratch: LSTM actor/twin
critic shape from legacy/recurrent_sac.py, and the dual-gate/capability-demotion
curriculum apparatus from curriculum_schedule.py and legacy/TD_MPC2.py, so results are
directly comparable to every other run in finalresults.md.

Replay buffer is split into a permanent `demo_buffer` (scripted-expert episodes, seeded
once, never evicted) and a growing `online_buffer`; every batch mixes DEMO_FRACTION from
the former. TD3+BC (Fujimoto & Gu 2021) adds an explicit imitation term to the actor loss
on top of that, after v33/v34 showed replay-buffer mixing alone wasn't enough to prevent
actor collapse (see git history / runs_registry.csv for v33-v36).

Usage (from repo root, PPO_IBM/):
    python legacy/TD3.py                 # fresh run, full curriculum
    python legacy/TD3.py --resume        # resume from latest checkpoint
"""

import os
import sys
import copy
import argparse
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "environments"))

from curriculum_starts import apply_saved_population, choose_episode_start
from training_state import find_latest_checkpoint, load_state, save_state
from curriculum_schedule import (
    ADVANCE_TARGETS, MASTERY_MIN_EPISODES, MASTERY_WINDOW, MASTERY_REQUIRED_STREAK,
    DEMOTION_CRASH_RATE, DEMOTION_STREAK_REQUIRED, CAPABILITY_DEMOTION_CHUNKS,
    _compute_curriculum_stats, _sample_training_difficulty,
)
from genetic_env import GeneticPhotobioreactorEnv

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
OBS_DIM = 6
ACTION_DIM = 3
MAX_CELLS = 7_500

# Matches PPO's gamma (recurrent_ppo.py Fix #13): harvest fires every 600 raw steps, so
# gamma needs an effective horizon (1/(1-gamma)) past that for credit assignment to reach it.
GAMMA = 0.9995

HIDDEN_DIM = 128
LSTM_LAYERS = 1
BATCH_SIZE = 24
SEQ_LEN = 60  # long enough for real pre/post context around a harvest event; see HARVEST_BIAS_PROB
ONLINE_BUFFER_CAPACITY = 5_000       # episodes
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
TAU = 0.005
GRAD_CLIP = 5.0
TRAIN_EVERY = 4  # gradient update every 4th env step; CPU-only, per-step updates measured too slow
POLICY_DELAY = 2                     # TD3: delayed actor + target updates
POLICY_NOISE = 0.2                   # target policy smoothing (Fujimoto et al. 2018 default)
NOISE_CLIP = 0.5

# TD3+BC actor regularization (Fujimoto & Gu 2021):
#   actor_loss = -lambda * Q1(s, pi(s)) + BC_COEF * MSE(pi(s_demo), a_demo)
#   lambda = TD3BC_ALPHA / mean(|Q1(s, pi(s))|).detach()  (normalizes Q-term to MSE's scale)
# BC term is evaluated on a fresh demo-only batch, not the mixed batch, so it never clones
# this run's own online actions.
TD3BC_ALPHA = 2.5    # paper default
BC_COEF = 1.0
EXPLORATION_NOISE_START = 0.25
EXPLORATION_NOISE_END = 0.03
EXPLORATION_NOISE_ANNEAL_FRAC = 0.3  # fraction of TOTAL_TRAINING_STEPS to anneal over

N_DEMO_EPISODES = 24                 # matches bc/bc_pretrain.py's default episode count
DEMO_FRACTION = 0.25                 # share of every training batch drawn from demos
DEMO_DIFFICULTY_WEIGHTS = {0: 0.4, 1: 0.4, 2: 0.2}  # matches bc/bc_pretrain.py

# Scripted-expert control law, numerically identical to bc/bc_pretrain.py (not imported,
# to avoid pulling in that module's SB3 dependency).
EXPERT_STIR_RANGE = (60.0, 80.0)
EXPERT_LIGHT_RANGE = (900.0, 1000.0)
EXPERT_OD_SETPOINT = 0.015
EXPERT_GAIN = 1.0
EXPERT_FRAC_CAP = 0.30

# Default budget is smaller than PPO/TD-MPC2's 8M-step convention: this file's per-step
# recurrent actor+twin-critic update measured well under 10 it/s on this CPU-only machine
# (>1 week for 8M steps). Override via TD3_STEPS. The dual-gate apparatus is chunk-based,
# so a smaller budget still produces a valid, honestly-reported outcome.
TOTAL_TRAINING_STEPS = int(os.environ.get("TD3_STEPS", "2000000"))
CHUNK_STEPS = 100_000
DET_EVAL_EPISODES_PER_CHUNK = 3
DET_MASTERY_MIN_EPISODES = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "model_data/td3_checkpoints"
STATE_PATH = "model_data/td3_training_state.pkl"
BUFFER_PATH = "model_data/td3_checkpoints/online_buffer.pkl"


# ═════════════════════════════════════════════════════════════════════════════
#  NETWORKS
# ═════════════════════════════════════════════════════════════════════════════

class RecurrentActor(nn.Module):
    """LSTM encoder + tanh-squashed deterministic head. Exploration noise is added
    externally during environment interaction, not sampled internally (unlike SAC)."""

    def __init__(self, obs_dim, action_dim, hidden_dim=HIDDEN_DIM, lstm_layers=LSTM_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.input_fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True)
        self.mean_fc = nn.Linear(hidden_dim, action_dim)

    def initial_hidden(self, batch):
        h = torch.zeros(self.lstm_layers, batch, self.hidden_dim, device=DEVICE)
        c = torch.zeros(self.lstm_layers, batch, self.hidden_dim, device=DEVICE)
        return (h, c)

    def forward(self, obs, hidden=None):
        """obs: [B, T, obs_dim]. Returns tanh-squashed action [B, T, action_dim] and hidden."""
        B, T, _ = obs.shape
        if hidden is None:
            hidden = self.initial_hidden(B)
        x = self.input_fc(obs.reshape(B * T, -1)).reshape(B, T, -1)
        out, hidden = self.lstm(x, hidden)
        action = torch.tanh(self.mean_fc(out))
        return action, hidden


class RecurrentCritic(nn.Module):
    """Twin independent-LSTM Q-networks (Fujimoto et al. 2018 twin-critic-min)."""

    def __init__(self, obs_dim, action_dim, hidden_dim=HIDDEN_DIM, lstm_layers=LSTM_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        inp = obs_dim + action_dim
        self.q1_input = nn.Sequential(nn.Linear(inp, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.q1_lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True)
        self.q1_head = nn.Linear(hidden_dim, 1)
        self.q2_input = nn.Sequential(nn.Linear(inp, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU())
        self.q2_lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True)
        self.q2_head = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch):
        zeros = lambda: torch.zeros(self.lstm_layers, batch, self.hidden_dim, device=DEVICE)
        return (zeros(), zeros()), (zeros(), zeros())

    def forward(self, obs, action, hidden1=None, hidden2=None):
        B, T, _ = obs.shape
        if hidden1 is None:
            hidden1, hidden2 = self.initial_hidden(B)
        sa = torch.cat([obs, action], dim=-1).reshape(B * T, -1)
        x1 = self.q1_input(sa).reshape(B, T, -1)
        x2 = self.q2_input(sa).reshape(B, T, -1)
        out1, hidden1 = self.q1_lstm(x1, hidden1)
        out2, hidden2 = self.q2_lstm(x2, hidden2)
        return self.q1_head(out1), self.q2_head(out2), hidden1, hidden2

    def q1_only(self, obs, action):
        B, T, _ = obs.shape
        hidden = (torch.zeros(self.lstm_layers, B, self.hidden_dim, device=DEVICE),
                  torch.zeros(self.lstm_layers, B, self.hidden_dim, device=DEVICE))
        sa = torch.cat([obs, action], dim=-1).reshape(B * T, -1)
        x = self.q1_input(sa).reshape(B, T, -1)
        out, _ = self.q1_lstm(x, hidden)
        return self.q1_head(out)


# ═════════════════════════════════════════════════════════════════════════════
#  SEQUENCE REPLAY BUFFER (episode-based, truncated-BPTT sampling)
# ═════════════════════════════════════════════════════════════════════════════

class SequenceReplayBuffer:
    # Harvest fires every HARVEST_INTERVAL_STEPS; uniform window sampling gives a
    # SEQ_LEN=25 window only ~4% odds of containing one at all (v33's collapse). Bias
    # sampling toward windows that include a harvest step.
    HARVEST_INTERVAL_STEPS = 600
    HARVEST_BIAS_PROB = 0.7  # rest sampled uniformly, for ordinary-transition coverage

    def __init__(self, capacity, seq_len):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = deque(maxlen=capacity)

    def add(self, episode):
        self.episodes.append(episode)

    def _sample_start(self, ep_len):
        if np.random.rand() < self.HARVEST_BIAS_PROB:
            candidates = list(range(self.HARVEST_INTERVAL_STEPS - 1, ep_len, self.HARVEST_INTERVAL_STEPS))
            if candidates:
                target = candidates[np.random.randint(len(candidates))]
                lo = max(0, target - self.seq_len + 1)
                hi = min(target, ep_len - self.seq_len)
                if hi >= lo:
                    return int(np.random.randint(lo, hi + 1))
        return int(np.random.randint(0, ep_len - self.seq_len + 1))

    def _sample_raw(self, batch_size):
        obs_b, act_b, rew_b, next_obs_b, done_b = [], [], [], [], []
        tries = 0
        while len(obs_b) < batch_size and tries < batch_size * 20:
            tries += 1
            if not self.episodes:
                break
            ep = self.episodes[np.random.randint(len(self.episodes))]
            ep_len = len(ep["reward"])
            if ep_len < self.seq_len:
                continue
            start = self._sample_start(ep_len)
            sl = slice(start, start + self.seq_len)
            obs_b.append(ep["obs"][sl]); act_b.append(ep["action"][sl])
            rew_b.append(ep["reward"][sl]); next_obs_b.append(ep["next_obs"][sl])
            done_b.append(ep["done"][sl])
        return obs_b, act_b, rew_b, next_obs_b, done_b

    def __len__(self):
        return len(self.episodes)

    def save(self, path):
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(list(self.episodes), fh)

    def load(self, path):
        import pickle
        if not os.path.exists(path):
            return
        with open(path, "rb") as fh:
            self.episodes = deque(pickle.load(fh), maxlen=self.capacity)


def sample_mixed_batch(demo_buffer, online_buffer, batch_size, demo_fraction):
    """demo_fraction of the batch from the permanent demo buffer, rest from online
    (falls back to all-demo if online can't fill its share yet)."""
    n_demo = int(round(batch_size * demo_fraction))
    n_online = batch_size - n_demo
    if len(online_buffer) == 0:
        n_demo, n_online = batch_size, 0

    d_obs, d_act, d_rew, d_nobs, d_done = demo_buffer._sample_raw(n_demo)
    o_obs, o_act, o_rew, o_nobs, o_done = online_buffer._sample_raw(n_online)

    obs = d_obs + o_obs; act = d_act + o_act; rew = d_rew + o_rew
    nobs = d_nobs + o_nobs; done = d_done + o_done
    if not obs:
        return None

    def to_t(lst, unsqueeze_last=False):
        t = torch.tensor(np.array(lst), dtype=torch.float32, device=DEVICE)
        return t.unsqueeze(-1) if unsqueeze_last else t

    return {
        "obs": to_t(obs), "action": to_t(act), "reward": to_t(rew, True),
        "next_obs": to_t(nobs), "done": to_t(done, True),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SCRIPTED-EXPERT DEMONSTRATION COLLECTION
# ═════════════════════════════════════════════════════════════════════════════

def expert_harvest_frac(od):
    surplus = (float(od) / EXPERT_OD_SETPOINT) - 1.0
    return float(np.clip(EXPERT_GAIN * surplus, 0.0, EXPERT_FRAC_CAP))


def collect_expert_demo_episode(difficulty, rng, seed):
    """One episode of the scripted proportional-harvest expert (same law validated in
    experiments/bc_scaffold/). Feeds the permanent demo_buffer."""
    from curriculum_schedule import _sample_init_cells
    init_cells = _sample_init_cells("random", difficulty)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    f_max = float(getattr(env, "F_MAX", 0.5))
    stir = float(rng.uniform(*EXPERT_STIR_RANGE))
    light = float(rng.uniform(*EXPERT_LIGHT_RANGE))

    ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []
    done = False
    while not done:
        frac = expert_harvest_frac(getattr(env, "od", 0.0))
        action = np.array([
            np.interp(stir, [50, 200], [-1, 1]),
            np.interp(light, [0, 2000], [-1, 1]),
            np.interp(frac, [0, f_max], [-1, 1]),
        ], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_obs.append(obs); ep_act.append(action); ep_rew.append(reward)
        ep_nobs.append(next_obs); ep_done.append(float(done))
        obs = next_obs

    return {
        "obs": np.array(ep_obs, dtype=np.float32), "action": np.array(ep_act, dtype=np.float32),
        "reward": np.array(ep_rew, dtype=np.float32), "next_obs": np.array(ep_nobs, dtype=np.float32),
        "done": np.array(ep_done, dtype=np.float32),
    }, {
        "harvested_mg": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
    }


def build_demo_buffer(n_episodes, seed=0):
    rng = np.random.default_rng(seed)
    diffs = list(DEMO_DIFFICULTY_WEIGHTS.keys())
    probs = np.array([DEMO_DIFFICULTY_WEIGHTS[d] for d in diffs], dtype=np.float64)
    probs /= probs.sum()

    buf = SequenceReplayBuffer(capacity=n_episodes, seq_len=SEQ_LEN)
    print(f"  Collecting {n_episodes} scripted-expert demonstration episodes "
          f"(mix {DEMO_DIFFICULTY_WEIGHTS})...")
    for i in range(n_episodes):
        difficulty = int(rng.choice(diffs, p=probs))
        episode, summary = collect_expert_demo_episode(difficulty, rng, seed=1_000_000 + i)
        buf.add(episode)
        print(f"    demo {i+1:>2}/{n_episodes}  D{difficulty}  "
              f"harvested={summary['harvested_mg']:7.1f}mg  time_avg_od={summary['time_avg_od']:.4f}")
    return buf


# ═════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC EVAL EPISODE (dual-gate — noise-free rollout)
# ═════════════════════════════════════════════════════════════════════════════

def run_td3_eval_episode(actor, difficulty, seed):
    """Noise-free rollout for the project's dual gate (see deterministic_eval.py /
    TD-MPC2's run_tdmpc2_eval_episode for the same rationale)."""
    from curriculum_schedule import _sample_init_cells
    init_cells = _sample_init_cells("random", difficulty)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    hidden = actor.initial_hidden(batch=1)
    done, step, info = False, 0, {}
    actor.eval()
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
            action_t, hidden = actor(obs_t, hidden)
            action = action_t.view(-1).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
    actor.train()
    return {
        "harvested_mg": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
        "crashed": step < env.max_steps,
        "start_mode": "low",
        "train_diff": difficulty,
        "reward": 0.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  TD3 UPDATE
# ═════════════════════════════════════════════════════════════════════════════

def soft_update(net, target_net, tau=TAU):
    for p, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def td3_update(actor, actor_target, critic, critic_target, actor_opt, critic_opt,
               demo_buffer, online_buffer, update_idx):
    batch = sample_mixed_batch(demo_buffer, online_buffer, BATCH_SIZE, DEMO_FRACTION)
    if batch is None:
        return None, None

    obs, actions, rewards = batch["obs"], batch["action"], batch["reward"]
    next_obs, dones = batch["next_obs"], batch["done"]

    with torch.no_grad():
        next_action, _ = actor_target(next_obs)
        noise = (torch.randn_like(next_action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (next_action + noise).clamp(-1.0, 1.0)
        q1_next, q2_next, _, _ = critic_target(next_obs, next_action)
        q_target = rewards + GAMMA * (1.0 - dones) * torch.min(q1_next, q2_next)

    q1, q2, _, _ = critic(obs, actions)
    # Huber, not MSE: genetic_env.py's crash penalty (-100) is a ~700x outlier against
    # typical per-step reward (experiments/env_diagnosis/), and GAMMA's long bootstrap
    # horizon spreads it across many Q-targets. Huber caps that outlier's gradient
    # contribution to linear instead of quadratic; identical to MSE for normal TD-errors.
    critic_loss = F.huber_loss(q1, q_target, delta=1.0) + F.huber_loss(q2, q_target, delta=1.0)
    critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
    critic_opt.step()

    actor_loss = None
    if update_idx % POLICY_DELAY == 0:
        pred_action, _ = actor(obs)
        q_pred = critic.q1_only(obs, pred_action)
        lam = torch.clamp(TD3BC_ALPHA / (q_pred.abs().mean().detach() + 1e-3), max=100.0)
        q_term = -lam * q_pred.mean()

        bc_obs, bc_act, _, _, _ = demo_buffer._sample_raw(BATCH_SIZE)
        bc_term = torch.tensor(0.0, device=DEVICE)
        if bc_obs:
            bc_obs_t = torch.tensor(np.array(bc_obs), dtype=torch.float32, device=DEVICE)
            bc_act_t = torch.tensor(np.array(bc_act), dtype=torch.float32, device=DEVICE)
            bc_pred, _ = actor(bc_obs_t)
            bc_term = BC_COEF * F.mse_loss(bc_pred, bc_act_t)

        actor_loss = q_term + bc_term
        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        actor_opt.step()

        soft_update(actor, actor_target)
        soft_update(critic, critic_target)
        actor_loss = float(actor_loss.item())

    return float(critic_loss.item()), actor_loss


# ═════════════════════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ═════════════════════════════════════════════════════════════════════════════

def save_checkpoint(actor, actor_target, critic, critic_target, actor_opt, critic_opt,
                    online_buffer, state):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(actor.state_dict(), f"{CHECKPOINT_DIR}/actor.pth")
    torch.save(actor_target.state_dict(), f"{CHECKPOINT_DIR}/actor_target.pth")
    torch.save(critic.state_dict(), f"{CHECKPOINT_DIR}/critic.pth")
    torch.save(critic_target.state_dict(), f"{CHECKPOINT_DIR}/critic_target.pth")
    torch.save(actor_opt.state_dict(), f"{CHECKPOINT_DIR}/actor_opt.pth")
    torch.save(critic_opt.state_dict(), f"{CHECKPOINT_DIR}/critic_opt.pth")
    online_buffer.save(BUFFER_PATH)
    save_state(STATE_PATH, state)


BEST_CHECKPOINT_DIR = "model_data/td3_checkpoints_best"


def save_best_checkpoint(actor, critic, det_harvest, global_step):
    """Separate, never-overwritten-by-collapse snapshot — the regular checkpoint only
    keeps the latest weights, so a later divergence can otherwise destroy the best
    result on disk. Overwrites only on genuine det_harvest improvement."""
    os.makedirs(BEST_CHECKPOINT_DIR, exist_ok=True)
    marker_path = f"{BEST_CHECKPOINT_DIR}/best_info.txt"
    prev_best = -1.0
    if os.path.exists(marker_path):
        try:
            with open(marker_path) as fh:
                prev_best = float(fh.read().split("det_harvest=")[1].split()[0])
        except Exception:
            prev_best = -1.0
    if det_harvest <= prev_best:
        return False
    torch.save(actor.state_dict(), f"{BEST_CHECKPOINT_DIR}/actor.pth")
    torch.save(critic.state_dict(), f"{BEST_CHECKPOINT_DIR}/critic.pth")
    with open(marker_path, "w") as fh:
        fh.write(f"step={global_step} det_harvest={det_harvest:.2f}\n")
    return True


def load_checkpoint(actor, actor_target, critic, critic_target, actor_opt, critic_opt, online_buffer):
    if not os.path.exists(f"{CHECKPOINT_DIR}/actor.pth"):
        return {}
    actor.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/actor.pth", map_location=DEVICE))
    actor_target.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/actor_target.pth", map_location=DEVICE))
    critic.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/critic.pth", map_location=DEVICE))
    critic_target.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/critic_target.pth", map_location=DEVICE))
    actor_opt.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/actor_opt.pth", map_location=DEVICE))
    critic_opt.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/critic_opt.pth", map_location=DEVICE))
    online_buffer.load(BUFFER_PATH)
    return load_state(STATE_PATH) or {}


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train(resume=False):
    from tqdm import tqdm

    print("--- TD3 (Twin Delayed DDPG) for GeneticPhotobioreactorEnv ---")
    print(f"Device: {DEVICE} | obs_dim={OBS_DIM} action_dim={ACTION_DIM} | gamma={GAMMA}")
    print(f"Budget: {TOTAL_TRAINING_STEPS:,} steps | Gate: project dual (stochastic + deterministic)")
    print(f"Demo replay: {N_DEMO_EPISODES} scripted-expert episodes, "
          f"{DEMO_FRACTION*100:.0f}% of every training batch, never evicted\n")

    actor = RecurrentActor(OBS_DIM, ACTION_DIM).to(DEVICE)
    actor_target = RecurrentActor(OBS_DIM, ACTION_DIM).to(DEVICE)
    actor_target.load_state_dict(actor.state_dict())
    critic = RecurrentCritic(OBS_DIM, ACTION_DIM).to(DEVICE)
    critic_target = RecurrentCritic(OBS_DIM, ACTION_DIM).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    for p in list(actor_target.parameters()) + list(critic_target.parameters()):
        p.requires_grad = False

    actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    online_buffer = SequenceReplayBuffer(ONLINE_BUFFER_CAPACITY, SEQ_LEN)

    saved_state = {}
    if resume:
        saved_state = load_checkpoint(actor, actor_target, critic, critic_target,
                                      actor_opt, critic_opt, online_buffer)
        if saved_state:
            print(f"  [RESUME] step={saved_state.get('global_step', 0):,} "
                  f"D{saved_state.get('current_difficulty', 0)} "
                  f"online_buffer={len(online_buffer)} episodes")
        else:
            print("  [RESUME] no checkpoint found, starting fresh.")

    # Regenerated deterministically (fixed seed) rather than persisted — pure function of
    # the scripted expert law, which never changes mid-run.
    demo_buffer = build_demo_buffer(N_DEMO_EPISODES, seed=0)

    global_step = saved_state.get("global_step", 0)
    current_difficulty = saved_state.get("current_difficulty", 0)
    mastery_streak = saved_state.get("mastery_streak", 0)
    demotion_streak = saved_state.get("demotion_streak", 0)
    capability_fail_streak = saved_state.get("capability_fail_streak", 0)
    completed_episodes = saved_state.get("completed_episodes", 0)
    saved_env_state = saved_state.get("saved_env_state", None)
    update_idx = saved_state.get("update_idx", 0)
    d0_capability_abort = False

    history_by_diff = defaultdict(lambda: deque(maxlen=MASTERY_WINDOW))
    for d, eps in saved_state.get("history_by_diff", {}).items():
        history_by_diff[d] = deque(eps, maxlen=MASTERY_WINDOW)
    det_eval_history = deque(saved_state.get("det_eval_history", []), maxlen=30)

    while global_step < TOTAL_TRAINING_STEPS and not d0_capability_abort:
        train_diff = _sample_training_difficulty(current_difficulty)
        start_cfg = choose_episode_start(train_diff, saved_state_available=saved_env_state is not None,
                                         completed_episodes=completed_episodes)
        init_cells = int(start_cfg["initial_cells"]) if start_cfg["initial_cells"] is not None else 3000
        chunk_steps = min(CHUNK_STEPS, TOTAL_TRAINING_STEPS - global_step)

        env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=train_diff)
        env.episode_start_mode = start_cfg["mode"]
        obs, _ = env.reset()
        if start_cfg["mode"] == "stitched" and saved_env_state is not None:
            apply_saved_population(env, saved_env_state)
            obs = env._get_obs()
        actor_hidden = actor.initial_hidden(batch=1)

        ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []
        episodes_this_chunk = 0

        print(f"\n[Chunk] train_diff=D{train_diff} | mastery_diff=D{current_difficulty} "
              f"| init={init_cells:,} | steps={chunk_steps:,}")
        pbar = tqdm(range(chunk_steps), desc=f"D{train_diff}", file=sys.stdout, mininterval=2.0)

        for _ in pbar:
            noise_frac = min(1.0, global_step / max(1, TOTAL_TRAINING_STEPS * EXPLORATION_NOISE_ANNEAL_FRAC))
            noise_scale = EXPLORATION_NOISE_START + noise_frac * (EXPLORATION_NOISE_END - EXPLORATION_NOISE_START)

            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
                action_t, actor_hidden = actor(obs_t, actor_hidden)
                action = action_t.view(-1).cpu().numpy()
            action = np.clip(action + np.random.normal(0, noise_scale, size=ACTION_DIM), -1.0, 1.0)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_obs.append(obs); ep_act.append(action); ep_rew.append(reward)
            ep_nobs.append(next_obs); ep_done.append(float(done))
            obs = next_obs
            global_step += 1

            if len(demo_buffer) > 0 and global_step % TRAIN_EVERY == 0:
                critic_loss, actor_loss = td3_update(actor, actor_target, critic, critic_target,
                                                     actor_opt, critic_opt, demo_buffer, online_buffer,
                                                     update_idx)
                update_idx += 1
                if global_step % 500 == 0 and critic_loss is not None:
                    pbar.set_postfix({"crit_loss": f"{critic_loss:.3f}",
                                      "noise": f"{noise_scale:.3f}", "Ep": episodes_this_chunk})

            if done:
                episodes_this_chunk += 1
                if len(ep_obs) >= SEQ_LEN:
                    online_buffer.add({
                        "obs": np.array(ep_obs, dtype=np.float32), "action": np.array(ep_act, dtype=np.float32),
                        "reward": np.array(ep_rew, dtype=np.float32), "next_obs": np.array(ep_nobs, dtype=np.float32),
                        "done": np.array(ep_done, dtype=np.float32),
                    })
                crashed = env.step_count < env.max_steps
                history_by_diff[train_diff].append({
                    "harvested_mg": float(info.get("cumulative_harvested_mg", 0.0)),
                    "time_avg_od": float(info.get("time_avg_od", 0.0)),
                    "crashed": crashed,
                    "start_mode": getattr(env, "episode_start_mode", "low"),
                    "train_diff": train_diff,
                    "reward": float(np.sum(ep_rew)) / max(len(ep_rew), 1),
                })
                completed_episodes += 1

                if getattr(env, "num_active", 0) > 15000:
                    saved_env_state = {
                        "cells_mass": copy.deepcopy(env.cells_mass), "cells_quota": copy.deepcopy(env.cells_quota),
                        "cells_x": copy.deepcopy(getattr(env, "cells_x", None)), "cells_z": copy.deepcopy(env.cells_z),
                        "clump_mass": copy.deepcopy(env.clump_mass), "pigment": env.pigment,
                        "num_active": env.num_active, "active_mask": copy.deepcopy(env.active_mask),
                        "ext_nutrients": env.ext_nutrients, "ph": env.ph, "do2": env.do2, "salt": env.salt,
                    }

                start_cfg = choose_episode_start(train_diff, saved_state_available=saved_env_state is not None,
                                                 completed_episodes=completed_episodes)
                init_cells = int(start_cfg["initial_cells"]) if start_cfg["initial_cells"] is not None else 3000
                env.initial_cells = init_cells
                env.episode_start_mode = start_cfg["mode"]
                obs, _ = env.reset()
                if start_cfg["mode"] == "stitched" and saved_env_state is not None:
                    apply_saved_population(env, saved_env_state)
                    obs = env._get_obs()
                actor_hidden = actor.initial_hidden(batch=1)
                ep_obs, ep_act, ep_rew, ep_nobs, ep_done = [], [], [], [], []

            if global_step % 50_000 == 0:
                save_checkpoint(actor, actor_target, critic, critic_target, actor_opt, critic_opt,
                                online_buffer, {
                                    "global_step": global_step, "current_difficulty": current_difficulty,
                                    "mastery_streak": mastery_streak, "demotion_streak": demotion_streak,
                                    "capability_fail_streak": capability_fail_streak,
                                    "completed_episodes": completed_episodes, "saved_env_state": saved_env_state,
                                    "update_idx": update_idx,
                                    "history_by_diff": {d: list(v) for d, v in history_by_diff.items()},
                                    "det_eval_history": list(det_eval_history),
                                })

        pbar.close()
        env.close()

        # Dual gate, same apparatus as legacy/TD_MPC2.py's Fix #15/#29 port.
        stats = _compute_curriculum_stats(list(history_by_diff[current_difficulty]), mastery_diff=current_difficulty)
        for i in range(DET_EVAL_EPISODES_PER_CHUNK):
            rec = run_td3_eval_episode(actor, current_difficulty, seed=100_000 + global_step + i)
            det_eval_history.append(rec)
        det_stats = _compute_curriculum_stats(list(det_eval_history), mastery_diff=current_difficulty)
        print(f"  [Det] eps={det_stats['episodes']} harvest_mg={det_stats['median_harvested_mg']:.1f} "
              f"p25={det_stats['p25_harvested_mg']:.1f} time_avg_od={det_stats['median_time_avg_od']:.4f} "
              f"crash={det_stats['crash_rate']*100:.1f}%")
        if det_stats["episodes"] >= DET_MASTERY_MIN_EPISODES and det_stats["crash_rate"] == 0.0:
            if save_best_checkpoint(actor, critic, det_stats["median_harvested_mg"], global_step):
                print(f"  [BEST] new best det checkpoint saved -> {BEST_CHECKPOINT_DIR} "
                      f"(harvest={det_stats['median_harvested_mg']:.1f}mg, 0% crash)")

        target = ADVANCE_TARGETS.get(current_difficulty)
        criteria_passed = det_criteria_passed = False
        if target is not None and stats["episodes"] >= MASTERY_MIN_EPISODES:
            criteria_passed = (stats["median_harvested_mg"] >= target["min_median_harvested_mg"]
                              and stats["p25_harvested_mg"] >= target["min_p25_harvested_mg"]
                              and stats["crash_rate"] <= target["max_crash_rate"]
                              and stats["median_time_avg_od"] >= target["min_median_time_avg_od"])
        if target is not None and det_stats["episodes"] >= DET_MASTERY_MIN_EPISODES:
            det_criteria_passed = (det_stats["median_harvested_mg"] >= target["min_median_harvested_mg"]
                                  and det_stats["p25_harvested_mg"] >= target["min_p25_harvested_mg"]
                                  and det_stats["crash_rate"] <= target["max_crash_rate"]
                                  and det_stats["median_time_avg_od"] >= target["min_median_time_avg_od"])
        criteria_passed = criteria_passed and det_criteria_passed

        next_difficulty = current_difficulty
        if stats["episodes"] >= MASTERY_MIN_EPISODES:
            if criteria_passed:
                mastery_streak += 1; demotion_streak = 0
            else:
                mastery_streak = 0
            if mastery_streak >= MASTERY_REQUIRED_STREAK:
                next_difficulty = min(2, current_difficulty + 1); mastery_streak = 0

            if current_difficulty > 0 and stats["crash_rate"] >= DEMOTION_CRASH_RATE:
                demotion_streak += 1
            else:
                demotion_streak = 0
            if demotion_streak >= DEMOTION_STREAK_REQUIRED:
                next_difficulty = max(0, current_difficulty - 1); mastery_streak = 0; demotion_streak = 0

            capability_failing = det_stats["episodes"] >= DET_MASTERY_MIN_EPISODES and not det_criteria_passed
            capability_fail_streak = capability_fail_streak + 1 if capability_failing else 0

            if capability_fail_streak >= CAPABILITY_DEMOTION_CHUNKS:
                if current_difficulty > 0:
                    print(f"  [CAPABILITY DEMOTION] deterministic gate failed {capability_fail_streak} "
                          f"consecutive chunks at D{current_difficulty} (crash rate "
                          f"{stats['crash_rate']:.2%}) — dropping a tier to restore a solvable task")
                    next_difficulty = max(0, current_difficulty - 1)
                    mastery_streak = demotion_streak = capability_fail_streak = 0
                elif stats["crash_rate"] >= DEMOTION_CRASH_RATE:
                    print(f"  [CAPABILITY ABORT] deterministic gate failed {capability_fail_streak} "
                          f"consecutive chunks at D0 (crash rate {stats['crash_rate']:.2%}) "
                          f"— no tier to demote to, stopping run.")
                    d0_capability_abort = True

        if next_difficulty != current_difficulty:
            direction = "ADVANCED" if next_difficulty > current_difficulty else "DEMOTED"
            print(f"  Curriculum {direction}: D{current_difficulty} -> D{next_difficulty} | "
                  f"chunk_eps={episodes_this_chunk} harvest_mg={stats['median_harvested_mg']:.1f} "
                  f"p25={stats['p25_harvested_mg']:.1f} time_avg_od={stats['median_time_avg_od']:.4f} "
                  f"crash={stats['crash_rate']:.2%}")
            capability_fail_streak = 0
        else:
            print(f"  Curriculum hold D{current_difficulty} | chunk_eps={episodes_this_chunk} "
                  f"eps={stats['episodes']} harvest_mg={stats['median_harvested_mg']:.1f} "
                  f"p25={stats['p25_harvested_mg']:.1f} time_avg_od={stats['median_time_avg_od']:.4f} "
                  f"crash={stats['crash_rate']:.2%} adv={mastery_streak}/{MASTERY_REQUIRED_STREAK} "
                  f"dem={demotion_streak}/{DEMOTION_STREAK_REQUIRED} "
                  f"capfail={capability_fail_streak}/{CAPABILITY_DEMOTION_CHUNKS}")
        current_difficulty = next_difficulty

    if d0_capability_abort:
        print("\n  [EARLY STOP] D0 capability-abort triggered — see log above.")
    print("\n--- Training Complete. Final model saved. ---")
    save_checkpoint(actor, actor_target, critic, critic_target, actor_opt, critic_opt, online_buffer, {
        "global_step": global_step, "current_difficulty": current_difficulty,
        "mastery_streak": mastery_streak, "demotion_streak": demotion_streak,
        "capability_fail_streak": capability_fail_streak, "completed_episodes": completed_episodes,
        "saved_env_state": saved_env_state, "update_idx": update_idx,
        "history_by_diff": {d: list(v) for d, v in history_by_diff.items()},
        "det_eval_history": list(det_eval_history),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 for GeneticPhotobioreactorEnv")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
