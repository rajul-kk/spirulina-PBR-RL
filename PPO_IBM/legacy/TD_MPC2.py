"""
TD-MPC2 (Temporal Difference Model Predictive Control) Implementation
For deeply-delayed, domain-randomized state-based control.
Upgrades: 1D-CNN History Compressor (24 steps), Policy Prior (Actor-Guided MPPI), Curriculum Learning (3 Phases).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from collections import deque
from curriculum_starts import apply_saved_population, choose_episode_start, mastery_metrics_view
from training_state import find_latest_checkpoint, load_state, replay_buffer_state, restore_replay_buffer, save_state

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ORDER = 16         # LMU memory depth
OBS_DIM      = 6   # Raw observation dimension
ACTION_DIM  = 4   # 4D action space [Stir, Light, Nutrient, CO2]
PRIV_DIM    = 4   # Privileged state dim [dissolved_co2, mean_fQ, mu_max, Ks_light]


class ObservationBuffer:
    """
    Holds the running LMU memory state `m_t`.
    No longer needs a full rolling window queue since LMU is continuous time.
    """
    def __init__(self, obs_dim: int = OBS_DIM, order: int = 16):
        self.obs_dim = obs_dim
        self.order = order
        # m_t shape: (1, OBS_DIM, ORDER) - batched for PyTorch ops
        self._m_t = None
        self.device = "cpu"

    def reset(self, obs: np.ndarray, device: str = "cpu"):
        """Initialize LMU state to zeros on environment reset."""
        self.device = device
        self._m_t = torch.zeros((1, self.obs_dim, self.order), device=device)

    def set_state(self, m_t: torch.Tensor):
        self._m_t = m_t

    def get_state(self) -> torch.Tensor:
        return self._m_t.clone() if self._m_t is not None else None


# ─── NETWORKS ─────────────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm — maps large sensor values to similar scale."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


class PrivilegedEncoder(nn.Module):
    """Maps privileged 4D state into latent space for optional distillation."""
    def __init__(self, priv_dim: int = PRIV_DIM, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(priv_dim, 128), nn.Mish(),
            nn.Linear(128, 128),      nn.Mish(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, priv: torch.Tensor) -> torch.Tensor:
        return self.net(priv)


class LMUHistoryCompressor(nn.Module):
    """
    Legendre Memory Unit (LMU): Compresses continuous observation history
    into a stateful encoding `m_t` and projects it to a 64D feature vector.

    Uses a fixed per-channel timescale (delta) for stable LMU dynamics.
    """
    def __init__(self, obs_dim: int = OBS_DIM, order: int = 16, init_theta: float = 250.0, out_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.order = order
        self.out_dim = out_dim
        
        # ── Initialize LMU Continuous-Time Matrices (A_base, B_base) ──
        # These are fixed matrices based on Legendre Polynomials
        Q = np.arange(order, dtype=float)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(j < i, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R

        # We store the pure continuous matrices. 
        # They will be scaled dynamically by the learned delta in the forward pass.
        self.register_buffer("A_base", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("B_base", torch.tensor(B, dtype=torch.float32))
        
        # Fixed per-channel timescale (Shape: OBS_DIM x 1)
        self.register_buffer("delta_fixed", torch.full((obs_dim, 1), 1.0 / init_theta, dtype=torch.float32))
        
        # ── Learnable Readout Network ──
        # Takes the raw observation (Dim) + the flattened LMU memory (Dim * Order)
        # and projects it to the 64D feature map the rest of the model expects.
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + (obs_dim * order), 256),
            nn.Mish(),
            nn.Linear(256, out_dim)
        )

    def forward(self, obs: torch.Tensor, m_t_minus_1: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        obs: (Batch, OBS_DIM)
        m_t_minus_1: (Batch, OBS_DIM, ORDER)   -- previous continuous memory state
        """
        batch_size = obs.shape[0] if obs.dim() > 1 else 1
        obs_sym = symlog(obs)
        
        # 1. Scale continuous matrices by the fixed timescale
        # delta shape: (obs_dim, 1)
        delta = self.delta_fixed.to(obs.device)
        
        # A_curr shape: (obs_dim, order, order)
        # B_curr shape: (obs_dim, order)
        # self.A_base is (16, 16). self.B_base is (16, 1)
        A_curr = self.A_base.unsqueeze(0) * delta.unsqueeze(-1)
        B_curr = self.B_base.transpose(0, 1) * delta

        # (Note: Using simple Forward Euler discretisation here for dynamic stability)
        # A_discrete = I + A_curr * dt (where dt=1 in simulation steps)
        # B_discrete = B_curr * dt
        device = obs.device
        I = torch.eye(self.order, device=device).unsqueeze(0)
        A_d = I + A_curr.to(device)
        B_d = B_curr.squeeze(-1).to(device)
        
        if m_t_minus_1 is None:
            m_t = torch.zeros((batch_size, self.obs_dim, self.order), device=device)
            # Deal with unbatched case for initial empty state
            if obs.dim() == 1:
                m_t = m_t.squeeze(0)
        else:
            # m_t_minus_1 shape: (BATCH, OBS_DIM, ORDER)
            # A shape: (ORDER, ORDER)
            # Deal with potential unbatched inputs (e.g. from single-step env interaction without unsqueeze)
            if m_t_minus_1.dim() == 2:
                # Shape: (OBS_DIM, ORDER) -> map inner J to outer Q with A(ORDER, ORDER)
                m_t_A = torch.einsum('dj,dqj->dq', m_t_minus_1, A_d)
                m_t_B = obs_sym * B_d
            else:
                # Shape: (BATCH, OBS_DIM, ORDER)
                m_t_A = torch.einsum('bdj,dqj->bdq', m_t_minus_1, A_d)
                m_t_B = obs_sym.unsqueeze(-1) * B_d.unsqueeze(0)
                
            m_t = m_t_A + m_t_B
            
        if m_t.dim() == 2:
            m_flat = m_t.reshape(-1)
        else:
            m_flat = m_t.reshape(batch_size, -1)
        # Readout uses both raw immediate obs and compressed history
        readout_input = torch.cat([obs_sym, m_flat], dim=-1)
        latent = self.fc(readout_input)
        
        return latent, m_t


class Encoder(nn.Module):
    """Maps 64D CNN embedding -> 64D Latent Abstract State (SimNorm output)."""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),   # always 64D from HistoryCompressor
            nn.Mish(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, obs):
        x = self.net(obs)
        # ── Simplicial Normalization (SimNorm) ──
        # Projects the unbounded latent vector onto a positive simplex.
        # This provides a bounded state space for the dynamics model, vastly 
        # improving sample efficiency and preventing exploding latents.
        x = F.elu(x) + 1.0  # Ensure strict positivity (using ELU to avoid dead neurons)
        return x / (x.norm(p=1, dim=-1, keepdim=True) + 1e-8)

class PolicyPrior(nn.Module):
    """
    Actor network: takes a latent state h and outputs a *mean* action.
    This biases the MPPI sampling N(pi(h), sigma) instead of N(0, sigma),
    focusing all 512 trajectories around the actor's best guess.
    Trained via behavioral cloning on the MPPI-chosen elite actions.
    """
    def __init__(self, latent_dim=64, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Keep output in [-1, 1]
        )

    def forward(self, h):
        return self.net(h)

class DynamicsModel(nn.Module):
    """Predicts Next Latent State: h_t+1 = D(h_t, a_t)"""
    def __init__(self, latent_dim=64, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, h, action):
        x = torch.cat([h, action], dim=-1)
        return self.net(x)

class RewardPredictor(nn.Module):
    """Predicts immediate reward from finding a state/action combo"""
    def __init__(self, latent_dim=64, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.Mish(),
            nn.Linear(256, 1)
        )
        
    def forward(self, h, action):
        x = torch.cat([h, action], dim=-1)
        return self.net(x)

class ValueNetwork(nn.Module):
    """Predicts expected future sum of rewards (Bootstrap)"""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Mish(),
            nn.Linear(256, 1)
        )
        
    def forward(self, h):
        return self.net(h)

# ─── TD-MPC2 AGENT ────────────────────────────────────────────────────────────

class TDMPC2Agent:
    def __init__(self, obs_dim: int, action_dim: int, device: str = "cpu",
                 use_privileged_distill: bool = False, distill_coef: float = 0.1):
        """obs_dim unused (kept for call-site compatibility). Encoder now takes 64D CNN output."""
        self.device = device
        self.latent_dim = 64

        # LMU history compressor (replaces CNN) + encoder
        self.compressor = LMUHistoryCompressor(obs_dim=OBS_DIM, order=16,
                                            out_dim=64).to(device)
        self.encoder        = Encoder(self.latent_dim).to(device)
        self.target_encoder = Encoder(self.latent_dim).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        self.dynamics = DynamicsModel(self.latent_dim, action_dim).to(device)
        self.reward_model = RewardPredictor(self.latent_dim, action_dim).to(device)

        # Policy Prior (Actor) — biases MPPI sampling distribution
        self.policy_prior = PolicyPrior(self.latent_dim, action_dim).to(device)

        # Twin Q-Networks for Value
        self.q1 = ValueNetwork(self.latent_dim).to(device)
        self.q2 = ValueNetwork(self.latent_dim).to(device)
        self.target_q1 = ValueNetwork(self.latent_dim).to(device)
        self.target_q2 = ValueNetwork(self.latent_dim).to(device)

        # Auxiliary Head: Predicts physical Population Density (OD) from latent space
        # Helps the CNN physically ground its random features
        self.aux_head = nn.Linear(self.latent_dim, 1).to(device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.encoder_opt = torch.optim.Adam(
            list(self.compressor.parameters()) + list(self.encoder.parameters()) + list(self.aux_head.parameters()), 
            lr=1e-3
        )
        self.dynamics_opt = torch.optim.Adam(self.dynamics.parameters(), lr=1e-3)
        self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=1e-3)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=1e-3)
        self.policy_opt = torch.optim.Adam(self.policy_prior.parameters(), lr=3e-4)

        # Optional privileged distillation path (toggleable)
        self.use_privileged_distill = use_privileged_distill
        self.distill_coef = distill_coef
        self.distil_step = 0
        if self.use_privileged_distill:
            self.priv_encoder = PrivilegedEncoder(priv_dim=PRIV_DIM, latent_dim=self.latent_dim).to(device)
            self.priv_opt = torch.optim.Adam(self.priv_encoder.parameters(), lr=3e-4)
        else:
            self.priv_encoder = None
            self.priv_opt = None

        self.action_dim = action_dim
        
    def _encode(self, obs: torch.Tensor, m_t: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, OBS_DIM) → 64D latent + updated LMU memory."""
        emb, new_m_t = self.compressor(obs, m_t)   # (B, 64), (B, OBS_DIM, ORDER)
        return self.encoder(emb), new_m_t         # (B, 64) with SimNorm

    def plan(self, obs: np.ndarray, m_t: torch.Tensor = None, horizon: int = 24,
             num_samples: int = 512, num_iters: int = 3) -> np.ndarray:
        """
        CEM/MPPI Planner with Policy Prior warm-start.
        obs: (OBS_DIM,) numpy array.
        m_t: Continuous latent state (OBS_DIM, ORDER).
        Returns the FIRST action of the optimal plan.
        """
        self.compressor.eval()
        self.encoder.eval()
        self.dynamics.eval()
        self.reward_model.eval()
        self.q1.eval()
        self.q2.eval()
        self.policy_prior.eval()

        with torch.no_grad():
            # 1. Encode single observation + memory via LMU compressor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, D)
            m_t_tensor = m_t.to(self.device) if m_t is not None else None
            h0, _ = self._encode(obs_tensor, m_t_tensor)  # [1, Latent]

            # 2. Policy Prior warm-start: bias the distribution mean
            # Without a prior: mean = zeros (blind search)
            # With a prior: mean = pi(h) (informed search around best guess)
            prior_action = self.policy_prior(h0)  # [1, ActionDim]
            # Expand mean across the horizon
            mean = prior_action.repeat(horizon, 1)  # [Horizon, ActionDim]
            std = torch.ones(horizon, self.action_dim).to(self.device) * 0.5  # Tighter search

            for _ in range(num_iters):
                # ── Differentiable CBF Bumper (Gradient Mean-Shift) ──
                # Temporarily enable gradients to compute analytical safety push
                with torch.enable_grad():
                    mean_opt = mean.clone().detach().requires_grad_(True)
                    h_mean = h0.clone()
                    safety_cost = 0.0
                    
                    for t in range(horizon):
                        a_t = mean_opt[t:t+1, :]
                        r_pred = self.reward_model(h_mean, a_t).squeeze(-1)
                        
                        # Accumulate penalty for predicted terminal/fatal states (r < 0)
                        safety_cost = safety_cost + torch.relu(-r_pred).sum()
                        
                        h_mean = self.dynamics(h_mean, a_t)
                    
                    # If current mean predicts danger, physically morph the trajectory
                    if safety_cost > 0.05:
                        grad_mean = torch.autograd.grad(safety_cost, mean_opt)[0]
                        # Shift the mean out of the danger zone (alpha = 0.1)
                        mean = (mean_opt - 0.1 * grad_mean).clamp(-1.0, 1.0).detach()
                # ─────────────────────────────────────────────────────

                # Sample: [Samples, Horizon, ActionDim] — biased around prior
                actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                    num_samples, horizon, self.action_dim, device=self.device
                )
                actions = torch.clamp(actions, -1.0, 1.0)

                # Rollout Dynamics
                h = h0.repeat(num_samples, 1)  # [Samples, Latent]
                returns = torch.zeros(num_samples, device=self.device)
                trajectory_rewards = torch.zeros(num_samples, horizon, device=self.device)

                for t in range(horizon):
                    a_t = actions[:, t, :]
                    reward = self.reward_model(h, a_t).squeeze(-1)
                    trajectory_rewards[:, t] = reward
                    returns += reward * (0.99 ** t)
                    
                    # ── L2 Action Smoothness Penalty (Jitter Tax) ──
                    if t > 0:
                        a_prev = actions[:, t-1, :]
                        # Penalize aggressive sudden shifts in Stirring/Light/Nutrients
                        smoothness_penalty = 0.025 * torch.sum((a_t - a_prev)**2, dim=-1)
                        returns -= smoothness_penalty * (0.99 ** t)

                    h = self.dynamics(h, a_t)

                # Terminal Value
                v1 = self.q1(h).squeeze(-1)
                v2 = self.q2(h).squeeze(-1)
                returns += torch.min(v1, v2) * (0.99 ** horizon)

                # ── The Latent CBF (Guillotine) ──
                # Cumulative sustainability check (Trajectory-wide)
                # If sum < 0, culture is net dying across the horizon
                trajectory_sums = trajectory_rewards.sum(dim=1)
                returns[trajectory_sums < 0.0] = -1e9

                # ── True Advantage-Weighted MPPI (Replacing Top-K) ──
                # baseline state value (from current state h0)
                baseline = torch.min(self.q1(h0), self.q2(h0)).squeeze(-1)
                advantage = returns - baseline
                
                kappa = 0.5  # Temperature parameter
                
                # Shift advantage for numerical stability in the exponential
                advantage_shifted = advantage - advantage.max()
                
                # Exponential weighting (Softmax)
                exp_weights = torch.exp(kappa * advantage_shifted)
                weights = exp_weights / (exp_weights.sum() + 1e-8)  # [Samples]
                
                # Reshape weights to broadcast across (Horizon, ActionDim)
                weights = weights.unsqueeze(-1).unsqueeze(-1)  # [Samples, 1, 1]
                
                # The new mean is the probability-weighted sum of ALL trajectories
                new_mean = (actions * weights).sum(dim=0)  # [Horizon, ActionDim]
                
                # Calculate standard deviation using the weights
                variance = ((actions - new_mean.unsqueeze(0))**2 * weights).sum(dim=0)
                new_std = torch.sqrt(variance).clamp(min=0.05)

                mean = 0.2 * mean + 0.8 * new_mean
                std = 0.2 * std + 0.8 * new_std

        best_action = mean[0].cpu().numpy()  # First action of best plan
        # Also cache the elite mean at t=0 for behavioral cloning the prior
        self._last_elite_action = mean[0].detach()
        return best_action
        
    def save(self, path):
        torch.save({
            'compressor': self.compressor.state_dict(),
            'encoder': self.encoder.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'dynamics': self.dynamics.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'policy_prior': self.policy_prior.state_dict(),
            'aux_head': self.aux_head.state_dict(),
            'use_privileged_distill': self.use_privileged_distill,
            'distill_coef': self.distill_coef,
            'priv_encoder': self.priv_encoder.state_dict() if self.priv_encoder is not None else None,
            'encoder_opt': self.encoder_opt.state_dict(),
            'dynamics_opt': self.dynamics_opt.state_dict(),
            'reward_opt': self.reward_opt.state_dict(),
            'q1_opt': self.q1_opt.state_dict(),
            'q2_opt': self.q2_opt.state_dict(),
            'policy_opt': self.policy_opt.state_dict(),
            'priv_opt': self.priv_opt.state_dict() if self.priv_opt is not None else None,
            'distil_step': self.distil_step,
        }, path)
        print(f"TD-MPC2 weights saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if 'compressor' in checkpoint:
            self.compressor.load_state_dict(checkpoint['compressor'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        if 'target_encoder' in checkpoint:
            self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        else:
            self.target_encoder.load_state_dict(checkpoint['encoder'])
        self.dynamics.load_state_dict(checkpoint['dynamics'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        if 'target_q1' in checkpoint:
            self.target_q1.load_state_dict(checkpoint['target_q1'])
        else:
            self.target_q1.load_state_dict(checkpoint['q1'])
        if 'target_q2' in checkpoint:
            self.target_q2.load_state_dict(checkpoint['target_q2'])
        else:
            self.target_q2.load_state_dict(checkpoint['q2'])
        if 'policy_prior' in checkpoint:
            self.policy_prior.load_state_dict(checkpoint['policy_prior'])
        if 'aux_head' in checkpoint:
            self.aux_head.load_state_dict(checkpoint['aux_head'])
        if self.priv_encoder is not None and checkpoint.get('priv_encoder') is not None:
            self.priv_encoder.load_state_dict(checkpoint['priv_encoder'])

        if 'encoder_opt' in checkpoint:
            self.encoder_opt.load_state_dict(checkpoint['encoder_opt'])
        if 'dynamics_opt' in checkpoint:
            self.dynamics_opt.load_state_dict(checkpoint['dynamics_opt'])
        if 'reward_opt' in checkpoint:
            self.reward_opt.load_state_dict(checkpoint['reward_opt'])
        if 'q1_opt' in checkpoint:
            self.q1_opt.load_state_dict(checkpoint['q1_opt'])
        if 'q2_opt' in checkpoint:
            self.q2_opt.load_state_dict(checkpoint['q2_opt'])
        if 'policy_opt' in checkpoint:
            self.policy_opt.load_state_dict(checkpoint['policy_opt'])
        if self.priv_opt is not None and checkpoint.get('priv_opt') is not None:
            self.priv_opt.load_state_dict(checkpoint['priv_opt'])
        self.distil_step = int(checkpoint.get('distil_step', self.distil_step))
        print(f"TD-MPC2 weights loaded from {path}")
        
    def update(self, batch_obs, batch_mt, batch_actions, batch_rewards, batch_next_obs, batch_next_mt, batch_dones,
               batch_priv=None):
        """
        Joint-Embedding Training Loop.
        batch_obs / batch_next_obs: (B, OBS_DIM) tensors.
        batch_mt / batch_next_mt: (B, OBS_DIM, ORDER) tensors.
        """
        self.compressor.train()
        self.encoder.train()
        self.dynamics.train()
        self.reward_model.train()
        self.q1.train()
        self.q2.train()
        self.policy_prior.train()

        obs      = torch.tensor(batch_obs,      dtype=torch.float32).to(self.device)
        mt       = torch.tensor(batch_mt,       dtype=torch.float32).to(self.device)
        actions  = torch.tensor(batch_actions,  dtype=torch.float32).to(self.device)
        rewards  = torch.tensor(batch_rewards,  dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_obs = torch.tensor(batch_next_obs, dtype=torch.float32).to(self.device)
        next_mt  = torch.tensor(batch_next_mt,  dtype=torch.float32).to(self.device)
        dones    = torch.tensor(batch_dones,    dtype=torch.float32).unsqueeze(-1).to(self.device)

        # 1. Target Encoding (No Gradients)
        with torch.no_grad():
            next_emb, _ = self.compressor(next_obs, next_mt)
            next_h_target = self.target_encoder(next_emb)
            next_q1 = self.target_q1(next_h_target)
            next_q2 = self.target_q2(next_h_target)
            # Apply (1 - dones) mask: bootstrap future Q ONLY if not terminal
            target_q = rewards + 0.99 * (1.0 - dones) * torch.min(next_q1, next_q2)

        # 2. Forward Pass
        h, _ = self._encode(obs, mt)
        pred_next_h = self.dynamics(h, actions)
        pred_reward = self.reward_model(h, actions)
        q1_pred = self.q1(h)
        q2_pred = self.q2(h)
        aux_pred = self.aux_head(h).squeeze(-1)

        # 3. World-Model Losses
        consistency_loss = F.mse_loss(pred_next_h, next_h_target)
        reward_loss = F.mse_loss(pred_reward, rewards)
        value_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        
        # Auxiliary Target: The physical population density (OD) is the 0th feature of next_obs
        # batch_next_obs is [B, OBS_DIM]. We want the 0th feature (OD).
        true_od = next_obs[:, 0] 
        aux_loss = F.mse_loss(aux_pred, true_od)

        total_loss = consistency_loss + reward_loss + value_loss + (0.1 * aux_loss)

        loss_distil = torch.tensor(0.0, device=self.device)
        teacher_loss = torch.tensor(0.0, device=self.device)
        if self.use_privileged_distill and batch_priv is not None and self.priv_encoder is not None:
            priv = torch.tensor(batch_priv, dtype=torch.float32).to(self.device)
            z_teacher = self.priv_encoder(priv)
            # Train teacher to regress current latent (stop-grad on student).
            teacher_loss = F.mse_loss(z_teacher, h.detach())
            # Distill student toward privileged teacher representation.
            loss_distil = F.mse_loss(h, z_teacher.detach())
            distil_weight = min(self.distil_step / 50_000.0, 1.0) * self.distill_coef
            total_loss = total_loss + distil_weight * loss_distil
            self.distil_step += 1

        # 4. Policy Prior Loss (Behavioral Cloning on actual env actions)
        # We supervise the Prior to predict the action the agent actually took.
        # Over time, 'actions' will increasingly be MPPI-elite actions,
        # so the Prior learns to warm-start the planner from real experience.
        prior_pred = self.policy_prior(h.detach())  # Detach: don't backprop into encoder twice
        policy_loss = F.mse_loss(prior_pred, actions)

        # 5. Optimize World Model
        self.encoder_opt.zero_grad()
        self.dynamics_opt.zero_grad()
        self.reward_opt.zero_grad()
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        total_loss.backward()
        self.encoder_opt.step()
        self.dynamics_opt.step()
        self.reward_opt.step()
        self.q1_opt.step()
        self.q2_opt.step()

        # 6. Optimize Policy Prior (separate pass)
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        if self.use_privileged_distill and self.priv_opt is not None and batch_priv is not None:
            self.priv_opt.zero_grad()
            teacher_loss.backward()
            self.priv_opt.step()

        # 7. Soft Update Target Networks (tau=0.01)
        # 7. Soft Update Target Networks (tau=0.01)
        tau = 0.01
        # Fix: Remove compressor (Source=Target) from zip loop to avoid redundant self-copy
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.q1.parameters(), self.target_q1.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.target_q2.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return {
            'loss/total': total_loss.item(),
            'loss/consistency': consistency_loss.item(),
            'loss/reward': reward_loss.item(),
            'loss/value': value_loss.item(),
            'loss/policy': policy_loss.item(),
            'loss/distil': loss_distil.item(),
        }

class ReplayBuffer:
    """Stores single observations and consecutive LMU states `m_t` instead of full history."""
    def __init__(self, capacity: int, obs_dim: int = OBS_DIM, action_dim: int = 4,
                 order: int = 16, priv_dim: int = PRIV_DIM):
        self.capacity = capacity
        self.obs       = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.m_t       = np.zeros((capacity, obs_dim, order), dtype=np.float32)
        self.actions   = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards   = np.zeros(capacity, dtype=np.float32)
        self.next_obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_m_t  = np.zeros((capacity, obs_dim, order), dtype=np.float32)
        self.dones     = np.zeros(capacity, dtype=np.float32)
        self.priv      = np.zeros((capacity, priv_dim), dtype=np.float32)
        self.idx  = 0
        self.size = 0

    def add(self, obs: np.ndarray, m_t: np.ndarray, action: np.ndarray, reward: float,
             next_obs: np.ndarray, next_m_t: np.ndarray, done: bool, priv: np.ndarray = None):
        self.obs[self.idx]       = obs
        self.m_t[self.idx]       = m_t
        self.actions[self.idx]   = action
        self.rewards[self.idx]   = reward
        self.next_obs[self.idx]  = next_obs
        self.next_m_t[self.idx]  = next_m_t
        self.dones[self.idx]     = done
        if priv is not None:
            self.priv[self.idx]  = priv
        self.idx  = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs],
            self.m_t[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.next_m_t[idxs],
            self.dones[idxs],
            self.priv[idxs],
        )

def train_td_mpc2(resume: bool = False, use_privileged_distill: bool = False):
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'environments'))
    from genetic_env import GeneticPhotobioreactorEnv
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    RAW_OBS_DIM = OBS_DIM
    ACTION_DIM = 4
    BATCH_SIZE = 512
    ACTION_REPEAT = 12
    PLANNING_HORIZON = 24
    MPPI_SAMPLES = 128

    TOTAL_TRAINING_STEPS = 1_500_000
    CHUNK_STEPS = 44_300
    MASTERY_WINDOW = 6
    MASTERY_MIN_EPISODES = 2
    MASTERY_REQUIRED_STREAK = 2
    ADVANCE_TARGETS = {
        0: {"median_od": 0.01, "min_p25_od": 0.007, "max_crash_rate": 0.05},
        1: {"median_od": 0.012, "min_p25_od": 0.009, "max_crash_rate": 0.05},
    }
    MIXING_PROBS = {
        0: ([0], [1.0]),
        1: ([1, 0], [0.8, 0.2]),
        2: ([2, 1, 0], [0.7, 0.2, 0.1]),
    }

    def _sample_training_difficulty(current_difficulty: int) -> int:
        diffs, probs = MIXING_PROBS[current_difficulty]
        return int(np.random.choice(diffs, p=probs))

    def _sample_init_cells(difficulty: int) -> int:
        if difficulty == 0 and np.random.rand() < 0.4:
            return 3000
        if difficulty == 2 and np.random.rand() < 0.1:
            return int(np.random.uniform(300, 800))
        return int(np.exp(np.random.uniform(np.log(1000), np.log(4000))))

    def _compute_curriculum_stats():
        if not recent_rewards:
            return {"episodes": 0, "median_od": 0.0, "crash_rate": 0.0, "reward_std": 0.0, "mean_pop": 0.0}
        return {
            "episodes": len(recent_rewards),
            "median_od": float(np.median(recent_ods)),
            "crash_rate": float(np.mean(recent_crashes)),
            "reward_std": float(np.std(recent_rewards)),
            "mean_pop": float(np.mean(recent_pops)),
        }

    def _advance_or_demote(current_difficulty: int, mastery_streak: int, stats: dict, train_diff: int):
        new_difficulty = current_difficulty
        new_streak = mastery_streak

        if stats["episodes"] < MASTERY_MIN_EPISODES:
            return new_difficulty, new_streak

        if current_difficulty in ADVANCE_TARGETS:
            target = ADVANCE_TARGETS[current_difficulty]
            passed = (
                stats["median_od"] >= target["median_od"]
                and stats["p25_od"] >= target["min_p25_od"]
                and stats["crash_rate"] <= target["max_crash_rate"]
            )
            if passed and train_diff == current_difficulty:
                new_streak += 1
                if new_streak >= MASTERY_REQUIRED_STREAK:
                    new_difficulty = min(2, current_difficulty + 1)
                    new_streak = 0
            elif not passed:
                new_streak = 0

        if current_difficulty > 0:
            baseline = ADVANCE_TARGETS.get(current_difficulty - 1, ADVANCE_TARGETS[0])["median_od"]
            severe_regression = (stats["crash_rate"] >= 0.20) or (stats["median_od"] < 0.5 * baseline)
            if severe_regression:
                new_difficulty = max(0, current_difficulty - 1)
                new_streak = 0

        return new_difficulty, new_streak

    print("--- Starting TD-MPC2 Adaptive Curriculum Training ---")
    print(f"LMU Order: {ORDER} state tracking | {OBS_DIM}D | MPPI Samples: {MPPI_SAMPLES}")

    checkpoint_dir = "model_data/tdmpc2_checkpoints"
    state_path = "model_data/tdmpc2_training_state.pkl"

    agent  = TDMPC2Agent(OBS_DIM, ACTION_DIM, device=device,
                         use_privileged_distill=use_privileged_distill)
    buffer = ReplayBuffer(25_000, OBS_DIM, ACTION_DIM)  # LMU continuous storage
    print(f"Privileged distillation: {'ON' if use_privileged_distill else 'OFF'}")

    recent_rewards = deque(maxlen=MASTERY_WINDOW)
    recent_pops    = deque(maxlen=MASTERY_WINDOW)
    recent_ods     = deque(maxlen=MASTERY_WINDOW)
    recent_crashes = deque(maxlen=MASTERY_WINDOW)
    chunk_metrics = []
    global_step = 0
    current_difficulty = 0
    mastery_streak = 0
    completed_episodes = 0
    has_restored_replay = False

    # ─── Resume Logic ───
    if resume:
        ckpt_path, ckpt_step = find_latest_checkpoint(checkpoint_dir, "tdmpc2_", "_steps.pth")
        state = load_state(state_path)
        if ckpt_path and state is not None:
            state_step = int(state.get("global_step", ckpt_step or 0))
            if ckpt_step is not None and abs(state_step - ckpt_step) > 2000:
                print(
                    f"  [CONTINUE] Checkpoint/state mismatch too large "
                    f"(ckpt={ckpt_step:,}, state={state_step:,}). "
                    f"Skipping resume to avoid inconsistent restore."
                )
            else:
                print(f"  [CONTINUE] Found latest checkpoint: {os.path.basename(ckpt_path)}")
                agent.load(ckpt_path)
                global_step = state_step
                current_difficulty = int(state.get("current_difficulty", 0))
                mastery_streak = int(state.get("mastery_streak", 0))
                completed_episodes = int(state.get("completed_episodes", 0))
                saved_env_state = state.get("saved_env_state")
                restore_replay_buffer(buffer, state.get("replay_buffer"))
                has_restored_replay = buffer.size > 0
                print(
                    f"  [CONTINUE] steps={global_step:,} | D{current_difficulty} | "
                    f"streak={mastery_streak} | buffer={buffer.size:,}"
                )
        else:
            print("  [CONTINUE] No matching checkpoint/state found. Starting from scratch.")

    raw_env = GeneticPhotobioreactorEnv(max_cells=300_000, initial_cells=3000, difficulty=2)
    obs_buf = ObservationBuffer(obs_dim=OBS_DIM, order=16)
    saved_env_state = locals().get("saved_env_state", None)

    if hasattr(raw_env, "set_difficulty"):
        raw_env.set_difficulty(0)
    else:
        raw_env.difficulty = 0
    raw_env.initial_cells = 3000

    raw_obs, _ = raw_env.reset()
    obs_buf.reset(raw_obs, device=device)
    obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
    _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
    obs_buf.set_state(m_t)

    if has_restored_replay:
        print(f"  [CONTINUE] Replay restored ({buffer.size:,} samples). Skipping random prefill.")
    else:
        print("  Pre-filling buffer with 2,000 random steps...")
        for _ in range(2000):
            action = raw_env.action_space.sample()
            next_raw_obs, reward, terminated, truncated, _ = raw_env.step(action)
            done = terminated or truncated

            cur_obs = raw_obs
            cur_mt = obs_buf.get_state()

            obs_tensor = torch.tensor(next_raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, next_m_t = agent.compressor(obs_tensor, cur_mt)

            obs_buf.set_state(next_m_t)
            priv_pre = raw_env.get_privileged_state() if (use_privileged_distill and hasattr(raw_env, 'get_privileged_state')) else None
            buffer.add(cur_obs, cur_mt.squeeze(0).cpu().detach().numpy(), action, reward,
                       next_raw_obs, next_m_t.squeeze(0).cpu().detach().numpy(), done, priv=priv_pre)
            raw_obs = next_raw_obs

            if done:
                raw_obs, _ = raw_env.reset()
                obs_buf.reset(raw_obs, device=device)
                obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
                obs_buf.set_state(m_t)

    while global_step < TOTAL_TRAINING_STEPS:
        train_diff = _sample_training_difficulty(current_difficulty)
        start_cfg = choose_episode_start(
            train_diff,
            saved_state_available=saved_env_state is not None,
            completed_episodes=completed_episodes,
        )
        init_cells = int(start_cfg["initial_cells"] if start_cfg["initial_cells"] is not None else 3000)
        chunk_steps = min(CHUNK_STEPS, TOTAL_TRAINING_STEPS - global_step)
        recent_rewards.clear()
        recent_pops.clear()
        recent_ods.clear()
        recent_crashes.clear()
        chunk_metrics.clear()

        if hasattr(raw_env, "set_difficulty"):
            raw_env.set_difficulty(train_diff)
        else:
            raw_env.difficulty = train_diff
        raw_env.initial_cells = init_cells
        raw_env.episode_start_mode = start_cfg["mode"]

        raw_obs, _ = raw_env.reset()
        if start_cfg["mode"] == "stitched" and saved_env_state is not None:
            apply_saved_population(raw_env, saved_env_state)
            raw_obs = raw_env._get_obs()
        obs_buf.reset(raw_obs, device=device)
        obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
        obs_buf.set_state(m_t)

        episode_reward = 0.0
        episode_peak_od = 0.0
        episodes_this_chunk = 0

        print(f"\n[Chunk] train_diff=D{train_diff} | mastery_diff=D{current_difficulty} | init={init_cells:,} | steps={chunk_steps:,}")
        pbar = tqdm(range(chunk_steps), desc=f"Adaptive D{train_diff}")

        for step in pbar:
            if step % ACTION_REPEAT == 0:
                action = agent.plan(raw_obs, obs_buf.get_state(), horizon=PLANNING_HORIZON,
                                    num_samples=MPPI_SAMPLES, num_iters=3)
                noise_scale = max(0.01, 0.15 * (1.0 - (global_step / 240_000)))
                action += np.random.normal(0, noise_scale, size=ACTION_DIM)
                action = np.clip(action, -1.0, 1.0)
                pbar.set_postfix({"Diff": f"D{train_diff}", "Mastery": f"D{current_difficulty}",
                                  "Stir": f"{action[0]:.2f}", "Light": f"{action[1]:.2f}", "Nutri": f"{action[2]:.2f}",
                                  "Ep": episodes_this_chunk, "OD": f"{np.mean(recent_ods):.4f}" if recent_ods else "--"})

            cur_obs = raw_obs
            cur_mt = obs_buf.get_state()

            next_raw_obs, reward, terminated, truncated, _ = raw_env.step(action)
            done = terminated or truncated

            obs_tensor = torch.tensor(next_raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, next_m_t = agent.compressor(obs_tensor, cur_mt)
            obs_buf.set_state(next_m_t)

            priv_state = raw_env.get_privileged_state() if (use_privileged_distill and hasattr(raw_env, 'get_privileged_state')) else None
            buffer.add(cur_obs, cur_mt.squeeze(0).cpu().detach().numpy(), action, reward,
                       next_raw_obs, next_m_t.squeeze(0).cpu().detach().numpy(), done, priv=priv_state)

            raw_obs = next_raw_obs
            episode_reward += reward
            episode_peak_od = max(episode_peak_od, getattr(raw_env, 'od', 0.0))
            global_step += 1

            if buffer.size >= BATCH_SIZE:
                ob_b, mt_b, act_b, rew_b, nob_b, nmt_b, done_b, priv_b = buffer.sample(BATCH_SIZE)
                agent.update(ob_b, mt_b, act_b, rew_b, nob_b, nmt_b, done_b, batch_priv=priv_b if use_privileged_distill else None)

            if done:
                episodes_this_chunk += 1
                ep_pop = getattr(raw_env, 'num_active', 0)
                recent_rewards.append(float(episode_reward))
                recent_pops.append(float(ep_pop))
                recent_ods.append(float(episode_peak_od))
                recent_crashes.append(float(ep_pop < 10))
                ep_len = int(getattr(raw_env, 'step_count', 0))
                reward_per_step = float(episode_reward) / max(ep_len, 1)
                chunk_metrics.append({
                    "reward": reward_per_step,
                    "peak_od": float(episode_peak_od),
                    "crashed": bool(ep_pop < 10),
                    "start_mode": getattr(raw_env, "episode_start_mode", "low"),
                })
                completed_episodes += 1
                mean_rew = np.mean(recent_rewards)
                mean_pop = np.mean(recent_pops)
                mean_od = np.mean(recent_ods)

                if ep_pop > 15000:
                    import copy
                    saved_env_state = {
                        'cells_mass': copy.deepcopy(raw_env.cells_mass),
                        'cells_quota': copy.deepcopy(raw_env.cells_quota),
                        'cells_x': copy.deepcopy(raw_env.cells_x),
                        'cells_z': copy.deepcopy(raw_env.cells_z),
                        'clump_mass': copy.deepcopy(raw_env.clump_mass),
                        'pigment': raw_env.pigment,
                        'num_active': raw_env.num_active,
                        'active_mask': copy.deepcopy(raw_env.active_mask),
                        'ext_nutrients': raw_env.ext_nutrients,
                        'ph': raw_env.ph,
                        'do2': raw_env.do2,
                        'salt': raw_env.salt,
                    }

                start_cfg = choose_episode_start(
                    train_diff,
                    saved_state_available=saved_env_state is not None,
                    completed_episodes=completed_episodes,
                )
                init_cells = int(start_cfg["initial_cells"] if start_cfg["initial_cells"] is not None else 3000)
                raw_env.initial_cells = init_cells
                raw_env.episode_start_mode = start_cfg["mode"]
                raw_obs, _ = raw_env.reset()
                if start_cfg["mode"] == "stitched" and saved_env_state is not None:
                    apply_saved_population(raw_env, saved_env_state)
                    raw_obs = raw_env._get_obs()
                obs_buf.reset(raw_obs, device=device)
                obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
                obs_buf.set_state(m_t)

                pbar.set_postfix({"Diff": f"D{train_diff}", "Mastery": f"D{current_difficulty}",
                                  "Ep": episodes_this_chunk, "Rew": f"{episode_reward:.1f}",
                                  "AvgRew": f"{mean_rew:.1f}", "PopAvg": f"{mean_pop:.0f}", "OD": f"{mean_od:.4f}",
                                  "Stir": f"{action[0]:.2f}", "Light": f"{action[1]:.2f}", "Nutri": f"{action[2]:.2f}"})
                episode_reward = 0.0
                episode_peak_od = 0.0

            if global_step % 2000 == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                agent.save(f"{checkpoint_dir}/tdmpc2_{global_step}_steps.pth")
                save_state(
                    state_path,
                    {
                        "global_step": global_step,
                        "current_difficulty": current_difficulty,
                        "mastery_streak": mastery_streak,
                        "completed_episodes": completed_episodes,
                        "saved_env_state": saved_env_state,
                        "replay_buffer": replay_buffer_state(
                            buffer,
                            ["obs", "m_t", "actions", "rewards", "next_obs", "next_m_t", "dones", "priv"],
                        ),
                    },
                )

        filtered_metrics = mastery_metrics_view(chunk_metrics)
        stats = {
            "episodes": len(filtered_metrics),
            "median_od": float(np.median([m["peak_od"] for m in filtered_metrics])) if filtered_metrics else 0.0,
            "p25_od": float(np.percentile([m["peak_od"] for m in filtered_metrics], 25)) if filtered_metrics else 0.0,
            "iqr_od": float(
                np.percentile([m["peak_od"] for m in filtered_metrics], 75)
                - np.percentile([m["peak_od"] for m in filtered_metrics], 25)
            ) if filtered_metrics else 0.0,
            "crash_rate": float(np.mean([1.0 if m["crashed"] else 0.0 for m in filtered_metrics])) if filtered_metrics else 0.0,
            "reward_std": float(np.std([m["reward"] for m in filtered_metrics])) if filtered_metrics else 0.0,
            "mean_pop": float(np.mean(recent_pops)) if recent_pops else 0.0,
        }
        target = ADVANCE_TARGETS.get(current_difficulty)
        criteria_passed = False
        if target is not None and stats["episodes"] >= MASTERY_MIN_EPISODES:
            criteria_passed = (
                stats["median_od"] >= target["median_od"]
                and stats["p25_od"] >= target["min_p25_od"]
                and stats["crash_rate"] <= target["max_crash_rate"]
            )

        # Inline streak tracking so the increment is explicit before the print
        next_difficulty = current_difficulty
        if stats["episodes"] >= MASTERY_MIN_EPISODES:
            if criteria_passed and train_diff == current_difficulty:
                mastery_streak += 1
            elif not criteria_passed:
                mastery_streak = 0
            # off-level pass: streak unchanged
            if mastery_streak >= MASTERY_REQUIRED_STREAK:
                next_difficulty = min(2, current_difficulty + 1)
                mastery_streak = 0
            elif current_difficulty > 0:
                baseline = ADVANCE_TARGETS.get(current_difficulty - 1, ADVANCE_TARGETS[0])["median_od"]
                severe_regression = (stats["crash_rate"] >= 0.20) or (stats["median_od"] < 0.5 * baseline)
                if severe_regression:
                    next_difficulty = max(0, current_difficulty - 1)
                    mastery_streak = 0

        if next_difficulty != current_difficulty:
            direction = "advanced" if next_difficulty > current_difficulty else "demoted"
            print(
                f"  Curriculum {direction}: D{current_difficulty} -> D{next_difficulty} "
                f"| chunk_eps={episodes_this_chunk}, median_OD={stats['median_od']:.4f}, "
                f"p25_OD={stats['p25_od']:.4f}, iqr_OD={stats['iqr_od']:.4f}, "
                f"crash={stats['crash_rate']:.2%}, reward_std(diag)={stats['reward_std']:.4f}"
            )
        else:
            if criteria_passed and train_diff != current_difficulty:
                streak_msg = f"pass(off-level D{train_diff})"
            else:
                streak_msg = f"{mastery_streak}/{MASTERY_REQUIRED_STREAK}"
            print(
                f"  Curriculum hold D{current_difficulty} | chunk_eps={episodes_this_chunk}, "
                f"eps={stats['episodes']}, median_OD={stats['median_od']:.4f}, p25_OD={stats['p25_od']:.4f}, "
                f"iqr_OD={stats['iqr_od']:.4f}, crash={stats['crash_rate']:.2%}, "
                f"reward_std(diag)={stats['reward_std']:.4f}, streak={streak_msg}"
            )
        current_difficulty = next_difficulty

    os.makedirs("model_data", exist_ok=True)
    agent.save("model_data/tdmpc2_genetic_ibm.pth")
    save_state(
        state_path,
        {
            "global_step": global_step,
            "current_difficulty": current_difficulty,
            "mastery_streak": mastery_streak,
            "completed_episodes": completed_episodes,
            "saved_env_state": saved_env_state,
            "replay_buffer": replay_buffer_state(
                buffer,
                ["obs", "m_t", "actions", "rewards", "next_obs", "next_m_t", "dones", "priv"],
            ),
        },
    )
    print("\n--- Training Complete. Final model saved. ---")


def finetune_td_mpc2(extra_steps: int = 500_000, use_privileged_distill: bool = False):
    """
    Continue TD-MPC2 training from a saved checkpoint at Difficulty 2 (Full Physics).
    Loads the world model, policy prior, and Q-network weights from the saved .pth file.
    Runs at a reduced exploration noise (0.05 vs 0.15) so the policy prior is trusted
    more heavily and MPPI focuses on refinement rather than exploration.
    """
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'environments'))
    from genetic_env import GeneticPhotobioreactorEnv
    from tqdm import tqdm

    model_path = "model_data/tdmpc2_genetic_ibm.pth"
    if not os.path.exists(model_path):
        print(f"  [ERROR] No saved model found at {model_path}")
        print("  Run 'python TD_MPC2.py' first to complete the curriculum.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    RAW_OBS_DIM   = OBS_DIM
    ACTION_DIM    = 4
    BATCH_SIZE    = 512
    ACTION_REPEAT = 12
    PLANNING_HORIZON = 24
    MPPI_SAMPLES  = 256

    print("─── TD-MPC2 Fine-Tune (Difficulty 2, Full Physics) ───")
    print(f"  Loading weights  : {model_path}")
    print(f"  Extra steps      : {extra_steps:,}")
    print(f"  Exploration noise: 0.05 (reduced from 0.15 — trust the prior)")

    agent = TDMPC2Agent(OBS_DIM, ACTION_DIM, device=device,
                        use_privileged_distill=use_privileged_distill)
    agent.load(model_path)
    print("  ✔ Weights loaded")

    buffer = ReplayBuffer(25_000, OBS_DIM, ACTION_DIM)

    raw_env = GeneticPhotobioreactorEnv(max_cells=300_000, initial_cells=3000, difficulty=2)
    obs_buf = ObservationBuffer(obs_dim=OBS_DIM, order=16)

    print("  Pre-filling buffer with 5,000 random steps...")
    raw_obs, _ = raw_env.reset()
    obs_buf.reset(raw_obs, device=device)
    obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
    _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
    obs_buf.set_state(m_t)
    
    for _ in range(5000):
        action = raw_env.action_space.sample()
        next_raw_obs, reward, terminated, truncated, _ = raw_env.step(action)
        done = terminated or truncated
        
        cur_obs = raw_obs
        cur_mt  = obs_buf.get_state()
        
        obs_tensor = torch.tensor(next_raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, next_m_t = agent.compressor(obs_tensor, cur_mt)
        obs_buf.set_state(next_m_t)
        
        buffer.add(cur_obs, cur_mt.squeeze(0).cpu().numpy(), action, reward,
               next_raw_obs, next_m_t.squeeze(0).cpu().numpy(), done)
                   
        raw_obs = next_raw_obs
        
        if done:
            raw_obs, _ = raw_env.reset()
            obs_buf.reset(raw_obs, device=device)
            obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
            obs_buf.set_state(m_t)

    raw_obs, _ = raw_env.reset()
    obs_buf.reset(raw_obs, device=device)
    obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
    _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
    obs_buf.set_state(m_t)
    
    episode_reward = 0.0
    episodes = 0
    pbar = tqdm(range(extra_steps), desc="Fine-Tune Phase 2")

    for step in pbar:
        if step % ACTION_REPEAT == 0:
            action = agent.plan(raw_obs, obs_buf.get_state(), horizon=PLANNING_HORIZON,
                                num_samples=MPPI_SAMPLES, num_iters=3)
            action += np.random.normal(0, 0.05, size=ACTION_DIM)
            action = np.clip(action, -1.0, 1.0)

        cur_obs = raw_obs
        cur_mt  = obs_buf.get_state()
        
        next_raw_obs, reward, terminated, truncated, _ = raw_env.step(action)
        done = terminated or truncated
        
        obs_tensor = torch.tensor(next_raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, next_m_t = agent.compressor(obs_tensor, cur_mt)
        obs_buf.set_state(next_m_t)
        
        buffer.add(cur_obs, cur_mt.squeeze(0).cpu().numpy(), action, reward,
               next_raw_obs, next_m_t.squeeze(0).cpu().numpy(), done)
                   
        raw_obs = next_raw_obs
        episode_reward += reward

        if buffer.size >= BATCH_SIZE:
            ob_b, mt_b, act_b, rew_b, nob_b, nmt_b, done_b, _ = buffer.sample(BATCH_SIZE)
            agent.update(ob_b, mt_b, act_b, rew_b, nob_b, nmt_b, done_b, batch_priv=None)

        if done:
            episodes += 1
            pbar.set_postfix({"Ep": episodes, "Pop": getattr(raw_env, 'num_active', 0),
                               "Rew": f"{episode_reward:.1f}",
                               "Stir": f"{action[0]:.2f}", "Light": f"{action[1]:.2f}", "Nutri": f"{action[2]:.2f}"})
            raw_obs, _ = raw_env.reset()
            obs_buf.reset(raw_obs, device=device)
            obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
            obs_buf.set_state(m_t)
            episode_reward = 0.0

        if step % 20_000 == 0 and step > 0:
            os.makedirs("model_data/tdmpc2_checkpoints", exist_ok=True)
            agent.save(f"model_data/tdmpc2_checkpoints/tdmpc2_finetune_{step}_steps.pth")

    agent.save(model_path)
    print(f"\n  Fine-tune complete. Model saved → {model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TD-MPC2 for GeneticPBR")
    parser.add_argument("--finetune", type=int, nargs='?', const=500_000,
                        help="Load saved model and continue training (optional: specify extra steps)")
    parser.add_argument("--resume", "--continue", dest="resume", action="store_true",
                        help="Continue curriculum training from the latest saved state.")
    parser.add_argument("--priv-distill", action="store_true",
                        help="Enable privileged distillation (training-time only).")
    args = parser.parse_args()

    if args.finetune is not None:
        finetune_td_mpc2(extra_steps=args.finetune, use_privileged_distill=args.priv_distill)
    else:
        train_td_mpc2(resume=args.resume, use_privileged_distill=args.priv_distill)
