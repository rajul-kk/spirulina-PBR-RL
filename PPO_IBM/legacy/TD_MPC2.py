"""
TD-MPC2 (Temporal Difference Model Predictive Control) Implementation
For deeply-delayed, domain-randomized state-based control.
Upgrades: 1D-CNN History Compressor (24 steps), Policy Prior (Actor-Guided MPPI), Curriculum Learning (3 Phases).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from collections import deque, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "environments"))

from curriculum_starts import apply_saved_population, choose_episode_start, mastery_metrics_view
from training_state import find_latest_checkpoint, load_state, replay_buffer_state, restore_replay_buffer, save_state
# Project curriculum gate — this file used to keep its own local ADVANCE_TARGETS keyed on
# median_od only. Rewired to the same gate PPO uses (harvest_mg / p25 / time_avg_od / crash)
# so a TD-MPC2 result is directly comparable to every PPO run in finalresults.md.
from curriculum_schedule import ADVANCE_TARGETS, MASTERY_MIN_EPISODES as PPO_MASTERY_MIN_EPISODES, _compute_curriculum_stats
from deterministic_eval import run_deterministic_eval_episode

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ORDER = 16         # LMU memory depth
OBS_DIM      = 6   # Raw observation dimension
# Fix (v27): action space was 4D [Stir, Light, Nutrient, CO2] — written against a pre-redesign
# env with manual CO2/nutrient dosing. The live env (genetic_env.py) has automated PID N/P
# dosing, no CO2 injection, and a 3D action space [stir, light, harvest]. This file could not
# construct the env at all with ACTION_DIM=4.
ACTION_DIM  = 3   # 3D action space [Stir, Light, Harvest] — matches genetic_env.py
PRIV_DIM    = 4   # Privileged state dim [dissolved_co2, mean_fQ, mu_max, Ks_light]

# Fix (v27): world-model MACRO-TIMESTEP. Each MPPI horizon step previously corresponded to one
# RAW env step (dt=0.02h), so horizon=24 saw only 0.48h ahead — the harvest event fires every
# HARVEST_INTERVAL_STEPS=600 raw steps (12h), so the planner was structurally blind to the one
# decision that has failed in every PPO run in this project (v4 through v24). Extending raw
# horizon to 600 was measured and rejected: cost scales ~linearly-to-superlinear with horizon,
# so h=600 vs h=24 projects to roughly 25x the already-measured 13h planning cost alone.
# Instead the dynamics/reward model is trained on MACRO-transitions spanning MACRO_STEPS raw
# steps (action held constant across the block, reward = discounted sum over the block). A
# planner horizon of 12 macro-steps then sees 12*MACRO_STEPS raw steps ahead. At MACRO_STEPS=50,
# horizon=12 -> 600 raw steps = exactly one harvest interval, at unchanged per-call planning
# cost. This also cuts the (measured, dominant) update() cost: replay stores ~1.5M/50=30,000
# macro-transitions instead of 1.5M raw ones, since update() is called once per macro-transition
# now rather than once per raw step.
MACRO_STEPS = 50

# Module-level (not local to train_td_mpc2) because TDMPC2Agent.update() also needs it for the
# block-length-adjusted Bellman bootstrap (GAMMA ** MACRO_STEPS) — a class method can't see a
# training-function-local variable.
GAMMA = 0.99

# Fix (v27): was 300_000 — measured directly (env.step() timing, not assumed) to cost 13.7ms/
# call vs 4.6ms/call at 7_500, a 3x per-step physics overhead, while ACTUAL active population
# in both cases never exceeded ~2,990 cells. max_cells is an array-allocation/masking cap, not
# something that changes physics outcomes below the cap, so 300_000 was buying nothing while
# tripling the dominant cost component (env.step() turned out to be far more expensive than
# plan()+update()+compressor combined — a cost this project's first TD-MPC2 measurement missed
# entirely by never timing env.step() in isolation). 7_500 matches max_cells everywhere else in
# this project (PPO's env_factory.py, all diagnostics), for direct comparability.
MAX_CELLS = 7_500


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


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class TwoHotEncoder:
    """Fix (v27): two-hot discrete regression for reward/value, replacing MSE — one of the two
    changes that distinguish TD-MPC2 from "MPC with a learned model" (the other is the Q
    ensemble below). Scalar targets are symlog-compressed, then represented as a two-hot
    vector over a fixed linear bin grid (mass split between the two bins bracketing the
    value, proportional to distance — exact if the value falls on a bin centre). The network
    predicts a categorical distribution over bins and is trained with cross-entropy; the
    scalar estimate is recovered as the expected bin value under that distribution.

    Why this over MSE: MSE regression on a wide-dynamic-range, heavy-tailed target (block
    rewards here range from near-zero to double digits depending on OD/harvest state) tends
    to be dominated by the largest-magnitude examples and gives no calibrated uncertainty.
    Two-hot classification is scale-robust by construction (symlog) and its softmax output
    is directly usable as a distributional value estimate.

    Verified with a standalone round-trip check (encode -> take the encoded distribution as
    if it were a perfect prediction -> decode) before being wired into training — see
    diagnostics/tdmpc2_cost_probe.py.
    """
    def __init__(self, vmin: float = -20.0, vmax: float = 20.0, num_bins: int = 101, device: str = "cpu"):
        self.num_bins = num_bins
        self.device = device
        self.bins = torch.linspace(vmin, vmax, num_bins, device=device)  # symlog-space bin centres
        self.bin_width = float(self.bins[1] - self.bins[0])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) raw scalar targets -> (B, num_bins) two-hot distribution."""
        x = symlog(x.reshape(-1))
        x = x.clamp(self.bins[0], self.bins[-1])
        idx_lo = torch.clamp(
            torch.floor((x - self.bins[0]) / self.bin_width).long(), 0, self.num_bins - 2
        )
        idx_hi = idx_lo + 1
        weight_hi = (x - self.bins[idx_lo]) / self.bin_width
        weight_lo = 1.0 - weight_hi
        out = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        out.scatter_(1, idx_lo.unsqueeze(-1), weight_lo.unsqueeze(-1))
        out.scatter_add_(1, idx_hi.unsqueeze(-1), weight_hi.unsqueeze(-1))
        return out

    def _expected_value(self, probs: torch.Tensor) -> torch.Tensor:
        """probs: (B, num_bins) already-normalised distribution -> (B,) scalar (symexp'd)."""
        symlog_val = (probs * self.bins.unsqueeze(0)).sum(dim=-1)
        return symexp(symlog_val)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (B, num_bins) RAW network output (not yet a distribution) -> (B,) scalar.
        Applies softmax first — do not call this on something already normalised (e.g. the
        output of encode()); use _expected_value directly for that, or the softmax will
        distort an already-valid distribution. This distinction is exactly what
        diagnostics/tdmpc2_cost_probe.py's round-trip test checks."""
        return self._expected_value(F.softmax(logits, dim=-1))


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
    """Predicts immediate (macro-block) reward from a state/action pair.
    Fix (v27): outputs num_bins logits (two-hot classification target) instead of 1 scalar
    (MSE target) — see TwoHotEncoder. Decoding to a scalar is the caller's responsibility
    (via TwoHotEncoder.decode), so this module stays a plain classifier head."""
    def __init__(self, latent_dim=64, action_dim=4, num_bins=101):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.Mish(),
            nn.Linear(256, num_bins)
        )

    def forward(self, h, action):
        x = torch.cat([h, action], dim=-1)
        return self.net(x)

class ValueNetwork(nn.Module):
    """Predicts expected future discounted return (Bootstrap). Fix (v27): num_bins logits,
    same two-hot rationale as RewardPredictor."""
    def __init__(self, latent_dim=64, num_bins=101):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Mish(),
            nn.Linear(256, num_bins)
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

        self.num_bins = 101
        # vmin/vmax are in SYMLOG units, not raw units — symlog(x)=sign(x)*log(|x|+1), so
        # +-20 symlog-units corresponds to raw values up to ~+-4.85e8. That range is standard
        # in the TD-MPC2/Dreamer literature because THEIR reward/value magnitudes reach into
        # the thousands; this project's per-block rewards and bootstrapped values are order
        # single-to-double-digits (measured: full-episode PPO rewards up to ~1120 over ~144
        # macro-blocks/episode -> ~5-8 per block, geometric Q-sum a low multiple of that). A
        # +-20 range would waste nearly all 101 bins on magnitudes never seen and give
        # terrible resolution exactly where this domain operates (verified directly: caught
        # by diagnostics/tdmpc2_cost_probe.py's round-trip test, which showed >3.0 raw-unit
        # decode error near symlog=3, i.e. raw~19, before this fix). +-6 symlog-units (raw
        # ~+-400) keeps generous headroom while giving ~4x finer resolution in-range.
        self.two_hot = TwoHotEncoder(vmin=-6.0, vmax=6.0, num_bins=self.num_bins, device=device)
        self.dynamics = DynamicsModel(self.latent_dim, action_dim).to(device)
        self.reward_model = RewardPredictor(self.latent_dim, action_dim, num_bins=self.num_bins).to(device)

        # Policy Prior (Actor) — biases MPPI sampling distribution
        self.policy_prior = PolicyPrior(self.latent_dim, action_dim).to(device)

        # Fix (v27): Q-ENSEMBLE, replacing the twin-Q pair. This is the second of the two
        # changes that make this genuinely "TD-MPC2" rather than "MPC with a learned model
        # and 2 critics" — the paper's ensemble (5 critics, random-subset-of-2 for the Bellman
        # target each update) reduces overestimation bias further than a fixed pair, since the
        # SAME two critics never get to collude with each other update after update.
        self.num_critics = 5
        self.qs = nn.ModuleList([
            ValueNetwork(self.latent_dim, num_bins=self.num_bins) for _ in range(self.num_critics)
        ]).to(device)
        self.target_qs = nn.ModuleList([
            ValueNetwork(self.latent_dim, num_bins=self.num_bins) for _ in range(self.num_critics)
        ]).to(device)
        for q, tq in zip(self.qs, self.target_qs):
            tq.load_state_dict(q.state_dict())

        # Auxiliary Head: Predicts physical Population Density (OD) from latent space
        # Helps the CNN physically ground its random features
        self.aux_head = nn.Linear(self.latent_dim, 1).to(device)

        # Optimizers
        self.encoder_opt = torch.optim.Adam(
            list(self.compressor.parameters()) + list(self.encoder.parameters()) + list(self.aux_head.parameters()),
            lr=1e-3
        )
        self.dynamics_opt = torch.optim.Adam(self.dynamics.parameters(), lr=1e-3)
        self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.q_opt = torch.optim.Adam(self.qs.parameters(), lr=1e-3)
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

    def _q_min_decoded(self, h: torch.Tensor, qs_list: nn.ModuleList, subset_size: int = 2) -> torch.Tensor:
        """Fix (v27): random-subset ensemble minimum (TD-MPC2's overestimation-reduction
        mechanism), decoded from two-hot logits to a scalar. A DIFFERENT random pair is drawn
        each call — including each call within the same plan()/update() — so no fixed pair of
        critics can collude with each other across updates the way a hard-coded twin-Q pair can.
        h: (B, latent_dim)."""
        idx = np.random.choice(len(qs_list), size=min(subset_size, len(qs_list)), replace=False)
        vals = torch.stack([self.two_hot.decode(qs_list[i](h)) for i in idx], dim=0)  # (subset, B)
        return vals.min(dim=0).values

    def _two_hot_ce_loss(self, logits: torch.Tensor, target_scalar: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against a two-hot soft target — the training-side counterpart to
        TwoHotEncoder.decode. Implemented explicitly (rather than relying on a specific
        PyTorch version's soft-label F.cross_entropy support) so behaviour is pinned regardless
        of torch version."""
        target_dist = self.two_hot.encode(target_scalar.reshape(-1))
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_dist * log_probs).sum(dim=-1).mean()

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
        self.qs.eval()
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
                        r_pred = self.two_hot.decode(self.reward_model(h_mean, a_t))

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
                    reward = self.two_hot.decode(self.reward_model(h, a_t))
                    trajectory_rewards[:, t] = reward
                    returns += reward * (0.99 ** t)
                    
                    # ── L2 Action Smoothness Penalty (Jitter Tax) ──
                    if t > 0:
                        a_prev = actions[:, t-1, :]
                        # Penalize aggressive sudden shifts in Stirring/Light/Nutrients
                        smoothness_penalty = 0.025 * torch.sum((a_t - a_prev)**2, dim=-1)
                        returns -= smoothness_penalty * (0.99 ** t)

                    h = self.dynamics(h, a_t)

                # Terminal Value — random-2-of-5 ensemble minimum (Fix v27)
                q_terminal = self._q_min_decoded(h, self.qs, subset_size=2)
                returns += q_terminal * (0.99 ** horizon)

                # ── The Latent CBF (Guillotine) ──
                # Cumulative sustainability check (Trajectory-wide)
                # If sum < 0, culture is net dying across the horizon
                trajectory_sums = trajectory_rewards.sum(dim=1)
                returns[trajectory_sums < 0.0] = -1e9

                # ── True Advantage-Weighted MPPI (Replacing Top-K) ──
                # baseline state value (from current state h0), ensemble minimum (Fix v27)
                baseline = self._q_min_decoded(h0, self.qs, subset_size=2)
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
            'qs': self.qs.state_dict(),
            'target_qs': self.target_qs.state_dict(),
            'policy_prior': self.policy_prior.state_dict(),
            'aux_head': self.aux_head.state_dict(),
            'use_privileged_distill': self.use_privileged_distill,
            'distill_coef': self.distill_coef,
            'priv_encoder': self.priv_encoder.state_dict() if self.priv_encoder is not None else None,
            'encoder_opt': self.encoder_opt.state_dict(),
            'dynamics_opt': self.dynamics_opt.state_dict(),
            'reward_opt': self.reward_opt.state_dict(),
            'q_opt': self.q_opt.state_dict(),
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
        # Fix (v27): 'qs'/'target_qs' is the current ModuleList format. Old checkpoints saved
        # under the twin-Q ('q1'/'q2') format cannot be loaded here — the network shapes
        # differ (num_bins-logit heads vs 1-scalar heads, 5 critics vs 2) — so this is a
        # deliberate hard break, not silently-wrong weights.
        if 'qs' in checkpoint:
            self.qs.load_state_dict(checkpoint['qs'])
            self.target_qs.load_state_dict(checkpoint.get('target_qs', checkpoint['qs']))
        elif 'q1' in checkpoint:
            raise RuntimeError(
                f"{path} was saved under the pre-Fix#27 twin-Q format (q1/q2, scalar heads) "
                f"and is incompatible with the current 5-critic two-hot ensemble. Retrain."
            )
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
        if 'q_opt' in checkpoint:
            self.q_opt.load_state_dict(checkpoint['q_opt'])
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
        self.qs.train()
        self.policy_prior.train()

        obs      = torch.tensor(batch_obs,      dtype=torch.float32).to(self.device)
        mt       = torch.tensor(batch_mt,       dtype=torch.float32).to(self.device)
        actions  = torch.tensor(batch_actions,  dtype=torch.float32).to(self.device)
        # Fix (v27): rewards are now per-MACRO-BLOCK discounted sums (see the training loop's
        # block_reward accumulation), not raw per-step rewards — MUCH wider dynamic range than
        # before, which is exactly the regime two-hot/symlog regression is meant for.
        rewards  = torch.tensor(batch_rewards,  dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(batch_next_obs, dtype=torch.float32).to(self.device)
        next_mt  = torch.tensor(batch_next_mt,  dtype=torch.float32).to(self.device)
        dones    = torch.tensor(batch_dones,    dtype=torch.float32).to(self.device)

        # 1. Target Encoding (No Gradients). Fix (v27): random-subset-of-2 ensemble minimum
        # from the 5 TARGET critics, decoded from two-hot logits, then re-encoded as the
        # two-hot classification target for ALL 5 online critics (standard ensemble Bellman
        # backup with random subsampling — TD-MPC2's overestimation-reduction mechanism).
        with torch.no_grad():
            next_emb, _ = self.compressor(next_obs, next_mt)
            next_h_target = self.target_encoder(next_emb)
            next_q_min = self._q_min_decoded(next_h_target, self.target_qs, subset_size=2)
            # GAMMA here is the per-MACRO-BLOCK discount — the block reward already folds in
            # GAMMA**t for t within the block, so bootstrapping the NEXT block needs GAMMA
            # raised to the block length once more, i.e. GAMMA_BLOCK = GAMMA ** MACRO_STEPS.
            target_q_scalar = rewards + (GAMMA ** MACRO_STEPS) * (1.0 - dones) * next_q_min

        # 2. Forward Pass
        h, _ = self._encode(obs, mt)
        pred_next_h = self.dynamics(h, actions)
        pred_reward_logits = self.reward_model(h, actions)
        aux_pred = self.aux_head(h).squeeze(-1)

        # 3. World-Model Losses. Fix (v27): reward/value are now two-hot cross-entropy, not MSE.
        consistency_loss = F.mse_loss(pred_next_h, next_h_target)
        reward_loss = self._two_hot_ce_loss(pred_reward_logits, rewards)
        value_loss = sum(
            self._two_hot_ce_loss(q(h), target_q_scalar) for q in self.qs
        ) / len(self.qs)

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
        self.q_opt.zero_grad()
        total_loss.backward()
        self.encoder_opt.step()
        self.dynamics_opt.step()
        self.reward_opt.step()
        self.q_opt.step()

        # 6. Optimize Policy Prior (separate pass)
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        if self.use_privileged_distill and self.priv_opt is not None and batch_priv is not None:
            self.priv_opt.zero_grad()
            teacher_loss.backward()
            self.priv_opt.step()

        # 7. Soft Update Target Networks (tau=0.01)
        tau = 0.01
        # Fix: Remove compressor (Source=Target) from zip loop to avoid redundant self-copy
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        # Fix (v27): all 5 critics soft-updated, not just a fixed twin pair.
        for p, tp in zip(self.qs.parameters(), self.target_qs.parameters()):
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

def run_tdmpc2_eval_episode(agent, difficulty, seed=None, horizon=12, num_samples=64):
    """Deterministic evaluation episode for the project's dual gate — mirrors
    deterministic_eval.run_deterministic_eval_episode's role and return shape, but for the
    TD-MPC2 agent's plan()/env interface (raw env + LMU memory, not SB3/VecNormalize).

    "Deterministic" here means running plan() WITHOUT the training loop's added exploration
    noise (`action += np.random.normal(...)`) — MPPI's own internal sampling is unavoidable,
    but the noise injected on top of the plan for exploration is not, and it is that
    exploration noise (not planner internals) that the dual gate exists to see past. Same
    project rationale as deterministic_eval.py: EpisodeMetricsCallback-equivalent stats come
    from noisy rollouts, and a policy that only "looks like" it works under exploration noise
    should not be able to advance on that alone.
    """
    from genetic_env import GeneticPhotobioreactorEnv
    from curriculum_schedule import _sample_init_cells
    if seed is not None:
        np.random.seed(seed)

    # Fix (v27 diagnostic): was hardcoded to 3000, giving this side of the gate a large,
    # policy-independent time_avg_od advantage over the stochastic side's curriculum-sampled
    # starts (100-1400 typical at D0) — a no-op policy alone clears D0's OD threshold at
    # init_cells=3000 (0.217 vs the 0.004 target). Sample the same way training does so both
    # sides of the dual gate are evaluated on comparable initial conditions.
    init_cells = _sample_init_cells("random", difficulty)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=difficulty)
    obs_buf = ObservationBuffer(obs_dim=OBS_DIM, order=16)
    raw_obs, _ = env.reset(seed=seed)
    obs_buf.reset(raw_obs, device=agent.device)
    obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
    obs_buf.set_state(m_t)

    done, step, info = False, 0, {}
    action = np.zeros(agent.action_dim, dtype=np.float32)
    while not done:
        if step % MACRO_STEPS == 0:
            action = agent.plan(raw_obs, obs_buf.get_state(), horizon=horizon,
                               num_samples=num_samples, num_iters=3)
        raw_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
        obs_buf.set_state(m_t)
        step += 1

    max_steps = env.max_steps
    return {
        "harvested_mg": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
        "crashed": step < max_steps,
        "start_mode": "low",
        "train_diff": difficulty,
        "reward": 0.0,
    }


def train_td_mpc2(resume: bool = False, use_privileged_distill: bool = False,
                  total_steps: int = None):
    import os
    from genetic_env import GeneticPhotobioreactorEnv
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 512
    # Fix (v27): ACTION_REPEAT == MACRO_STEPS by construction — the plan() replan cadence and
    # the world-model macro-transition size are the SAME thing. Replanning more often than the
    # model's own timestep resolution would be replanning on stale/unlearned dynamics; less
    # often would waste the model's resolution. horizon=12 * MACRO_STEPS=50 = 600 raw steps =
    # exactly one HARVEST_INTERVAL_STEPS, so the planner can now see across a harvest event.
    ACTION_REPEAT = MACRO_STEPS
    PLANNING_HORIZON = 12
    MPPI_SAMPLES = 64

    # Fix (v27): TOTAL_TRAINING_STEPS was 1_500_000 (~1.9 days measured pre-macro-transition
    # cost, dominated by update() at 1 call/raw-step). With macro-transitions update() now
    # fires once per MACRO_STEPS raw steps, so cost drops accordingly — re-measured before this
    # number is trusted (see diagnostics/tdmpc2_cost_probe.py). Overridable via total_steps= /
    # TDMPC2_STEPS env var so a run's budget doesn't require a source edit.
    # 8,000,000 matches every PPO run's budget in finalresults.md, for direct comparability.
    # Measured cost at this budget (diagnostics/tdmpc2_cost_probe.py): ~13.0h, under PPO's
    # own ~17h — affordable now that macro-transitions cut the update()-call count ~30x.
    TOTAL_TRAINING_STEPS = total_steps or int(os.environ.get("TDMPC2_STEPS", "8000000"))
    CHUNK_STEPS = 100_000  # matches recurrent_ppo.py's CHUNK_STEPS convention
    # Fix (v27): mastery window/streak now match curriculum_schedule.py's PPO defaults so a
    # TD-MPC2 run's gate is not just the SAME metric as PPO's, it fires on the SAME cadence.
    from curriculum_schedule import MASTERY_WINDOW, MASTERY_REQUIRED_STREAK
    MASTERY_MIN_EPISODES = PPO_MASTERY_MIN_EPISODES
    DET_EVAL_EPISODES_PER_CHUNK = 3
    DET_MASTERY_MIN_EPISODES = 9
    MIXING_PROBS = {
        0: ([0], [1.0]),
        1: ([1, 0], [0.8, 0.2]),
        2: ([2, 1, 0], [0.7, 0.2, 0.1]),
    }

    def _sample_training_difficulty(current_difficulty: int) -> int:
        diffs, probs = MIXING_PROBS[current_difficulty]
        return int(np.random.choice(diffs, p=probs))

    print("--- Starting TD-MPC2 Adaptive Curriculum Training ---")
    print(f"LMU Order: {ORDER} state tracking | {OBS_DIM}D obs | {ACTION_DIM}D action "
          f"| MPPI samples={MPPI_SAMPLES} horizon={PLANNING_HORIZON} macro_steps={MACRO_STEPS} "
          f"(-> {PLANNING_HORIZON * MACRO_STEPS} raw steps of lookahead)")
    print(f"Budget: {TOTAL_TRAINING_STEPS:,} steps | Gate: project dual (stochastic + deterministic)")

    checkpoint_dir = "model_data/tdmpc2_checkpoints"
    state_path = "model_data/tdmpc2_training_state.pkl"

    agent  = TDMPC2Agent(OBS_DIM, ACTION_DIM, device=device,
                         use_privileged_distill=use_privileged_distill)
    # Capacity in MACRO-transitions now, not raw steps — each entry already spans MACRO_STEPS
    # raw steps, so 25,000 macro-transitions covers 1.25M raw steps of experience, comparable
    # coverage to the old raw-step buffer at a fraction of the memory.
    buffer = ReplayBuffer(25_000, OBS_DIM, ACTION_DIM)
    print(f"Privileged distillation: {'ON' if use_privileged_distill else 'OFF'}")

    chunk_metrics = []          # stochastic rollout episodes this chunk (PPO-shaped dicts)
    det_eval_history = deque(maxlen=30)  # deterministic eval episodes, project-shaped dicts
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

    raw_env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=3000, difficulty=2)
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
        # Fix (v27): prefill also produces MACRO-transitions (random action held for
        # MACRO_STEPS raw steps each), not raw single-step transitions. Mixing 1-raw-step and
        # 50-raw-step transitions in the same buffer would teach the dynamics model two
        # different, contradictory timestep resolutions.
        n_prefill_blocks = 2000 // MACRO_STEPS
        print(f"  Pre-filling buffer with {n_prefill_blocks} macro-transitions "
              f"({n_prefill_blocks * MACRO_STEPS} raw steps)...")
        for _ in range(n_prefill_blocks):
            action = raw_env.action_space.sample()
            block_start_obs, block_start_mt = raw_obs, obs_buf.get_state()
            block_reward, block_discount, done = 0.0, 1.0, False
            for _ in range(MACRO_STEPS):
                next_raw_obs, reward, terminated, truncated, _ = raw_env.step(action)
                done = terminated or truncated
                block_reward += reward * block_discount
                block_discount *= GAMMA
                obs_tensor = torch.tensor(next_raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, next_m_t = agent.compressor(obs_tensor, obs_buf.get_state())
                obs_buf.set_state(next_m_t)
                raw_obs = next_raw_obs
                if done:
                    break

            priv_pre = raw_env.get_privileged_state() if (use_privileged_distill and hasattr(raw_env, 'get_privileged_state')) else None
            buffer.add(block_start_obs, block_start_mt.squeeze(0).cpu().detach().numpy(), action,
                      block_reward, raw_obs, obs_buf.get_state().squeeze(0).cpu().detach().numpy(),
                      done, priv=priv_pre)

            if done:
                raw_obs, _ = raw_env.reset()
                obs_buf.reset(raw_obs, device=device)
                obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, m_t = agent.compressor(obs_tensor, obs_buf.get_state())
                obs_buf.set_state(m_t)

    # Persistent per-difficulty rolling episode history — mirrors recurrent_ppo.py's
    # EpisodeMetricsCallback (metrics_cb.history_by_diff), NOT the original file's
    # chunk-local `chunk_metrics.clear()` — a chunk-local window under-samples badly here:
    # CHUNK_STEPS=100,000 raw steps / MACRO_STEPS=50 ~= 2,000 macro-decisions/chunk, and a
    # ~7200-step episode is ~144 macro-transitions, so a chunk holds only ~14 episodes —
    # below MASTERY_MIN_EPISODES=20 on its own. A persistent window (matching PPO's
    # MASTERY_WINDOW=40) lets the gate accumulate across chunk boundaries the same way PPO's
    # does, rather than resetting evidence every chunk.
    history_by_diff = defaultdict(lambda: deque(maxlen=MASTERY_WINDOW))
    demotion_streak = 0

    while global_step < TOTAL_TRAINING_STEPS:
        train_diff = _sample_training_difficulty(current_difficulty)
        start_cfg = choose_episode_start(
            train_diff,
            saved_state_available=saved_env_state is not None,
            completed_episodes=completed_episodes,
        )
        init_cells = int(start_cfg["initial_cells"] if start_cfg["initial_cells"] is not None else 3000)
        chunk_steps = min(CHUNK_STEPS, TOTAL_TRAINING_STEPS - global_step)

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
        episodes_this_chunk = 0
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        print(f"\n[Chunk] train_diff=D{train_diff} | mastery_diff=D{current_difficulty} | init={init_cells:,} | steps={chunk_steps:,}")
        pbar = tqdm(range(chunk_steps), desc=f"Adaptive D{train_diff}")

        # Macro-transition accumulator state (reset at the top of every new block, i.e. every
        # MACRO_STEPS raw steps, or immediately after an episode ends mid-block).
        block_start_obs, block_start_mt = raw_obs, obs_buf.get_state()
        block_reward, block_discount = 0.0, 1.0
        # Forces a fresh plan()+block at the NEXT loop iteration regardless of the raw
        # step % MACRO_STEPS alignment. Needed after an episode reset: `step` is the
        # CHUNK-level counter and does not reset per episode, so without this an episode
        # could start mid-way through what the accumulator thinks is an old block, applying
        # a stale action (planned for the previous episode's last state) to a brand-new one.
        force_new_block = False

        def _start_block(cur_raw_obs, cur_step):
            act = agent.plan(cur_raw_obs, obs_buf.get_state(), horizon=PLANNING_HORIZON,
                             num_samples=MPPI_SAMPLES, num_iters=3)
            noise_scale = max(0.01, 0.15 * (1.0 - (cur_step / (TOTAL_TRAINING_STEPS * 0.3))))
            act += np.random.normal(0, noise_scale, size=ACTION_DIM)
            return np.clip(act, -1.0, 1.0)

        for step in pbar:
            new_block = (step % MACRO_STEPS == 0) or force_new_block
            if new_block:
                force_new_block = False
                block_start_obs, block_start_mt = raw_obs, obs_buf.get_state()
                block_reward, block_discount = 0.0, 1.0
                action = _start_block(raw_obs, global_step)
                pbar.set_postfix({"Diff": f"D{train_diff}", "Mastery": f"D{current_difficulty}",
                                  "Stir": f"{action[0]:.2f}", "Light": f"{action[1]:.2f}", "Harv": f"{action[2]:.2f}",
                                  "Ep": episodes_this_chunk})

            next_raw_obs, reward, terminated, truncated, info = raw_env.step(action)
            done = terminated or truncated
            block_reward += reward * block_discount
            block_discount *= GAMMA

            # LMU memory updates every RAW step regardless of macro cadence — it is the fine-
            # grained sensor-history compressor, not the coarse world-model transition unit.
            obs_tensor = torch.tensor(next_raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, next_m_t = agent.compressor(obs_tensor, obs_buf.get_state())
            obs_buf.set_state(next_m_t)
            raw_obs = next_raw_obs
            episode_reward += reward
            global_step += 1

            block_end = ((step + 1) % MACRO_STEPS == 0) or done
            if block_end:
                priv_state = raw_env.get_privileged_state() if (use_privileged_distill and hasattr(raw_env, 'get_privileged_state')) else None
                buffer.add(block_start_obs, block_start_mt.squeeze(0).cpu().detach().numpy(), action,
                          block_reward, raw_obs, obs_buf.get_state().squeeze(0).cpu().detach().numpy(),
                          done, priv=priv_state)
                if buffer.size >= BATCH_SIZE:
                    ob_b, mt_b, act_b, rew_b, nob_b, nmt_b, done_b, priv_b = buffer.sample(BATCH_SIZE)
                    agent.update(ob_b, mt_b, act_b, rew_b, nob_b, nmt_b, done_b, batch_priv=priv_b if use_privileged_distill else None)

            if done:
                episodes_this_chunk += 1
                # Fix (v27): read the PROJECT's own harvest_mg / time_avg_od metrics from the
                # step info dict (genetic_env.py always populates these) rather than the
                # original file's ad-hoc "peak_od" / "population < 10" proxies. This is what
                # makes stats directly comparable against ADVANCE_TARGETS and every PPO run.
                crashed = bool(getattr(raw_env, "step_count", 0) < raw_env.max_steps)
                ep_len = int(getattr(raw_env, 'step_count', 0))
                reward_per_step = float(episode_reward) / max(ep_len, 1)
                history_by_diff[train_diff].append({
                    "harvested_mg": float(info.get("cumulative_harvested_mg", 0.0)),
                    "time_avg_od": float(info.get("time_avg_od", 0.0)),
                    "crashed": crashed,
                    "start_mode": getattr(raw_env, "episode_start_mode", "low"),
                    "train_diff": train_diff,
                    "reward": reward_per_step,
                })
                completed_episodes += 1
                ep_pop = getattr(raw_env, 'num_active', 0)

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
                # See the force_new_block comment above the loop: `step` does not reset per
                # episode, so this is what makes the NEXT iteration replan on the fresh state
                # instead of continuing a block that belonged to the episode that just ended.
                force_new_block = True

                last_rec = history_by_diff[train_diff][-1] if history_by_diff[train_diff] else {}
                pbar.set_postfix({"Diff": f"D{train_diff}", "Mastery": f"D{current_difficulty}",
                                  "Ep": episodes_this_chunk, "Rew": f"{episode_reward:.1f}",
                                  "Harv_mg": f"{last_rec.get('harvested_mg', 0.0):.1f}",
                                  "OD": f"{last_rec.get('time_avg_od', 0.0):.4f}",
                                  "Stir": f"{action[0]:.2f}", "Light": f"{action[1]:.2f}", "Harv": f"{action[2]:.2f}"})
                episode_reward = 0.0

            # Fix (v27 diagnostic): was every 2,000 raw steps (4,000 saves over an 8M-step
            # budget), each one pickling the FULL 25,000-transition replay buffer on top of
            # network weights — ~7MB/save, ~15GB and rising over the v27 run, and a plausible
            # contributor to the chunk-time variance observed all session under CPU contention.
            # PPO's CheckpointCallback saves every 10,000 steps and weights only (no persistent
            # buffer to dump). Widened 25x; still ~320 saves over the full budget.
            if global_step % 50_000 == 0:
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

        # ── Fix (v27): dual gate — project's ADVANCE_TARGETS + deterministic eval ──────────
        # Stochastic side: persistent rolling history for the CURRENT mastery difficulty.
        stats = _compute_curriculum_stats(list(history_by_diff[current_difficulty]),
                                          mastery_diff=current_difficulty)

        # Deterministic side: a handful of noise-free planning episodes per chunk, same
        # project rationale as recurrent_ppo.py's det_eval_history — a policy that only
        # "looks like" it works under exploration noise should not be able to advance alone.
        for _ in range(DET_EVAL_EPISODES_PER_CHUNK):
            rec = run_tdmpc2_eval_episode(agent, current_difficulty,
                                          seed=100_000 + global_step + _,
                                          horizon=PLANNING_HORIZON, num_samples=MPPI_SAMPLES)
            det_eval_history.append(rec)
        det_stats = _compute_curriculum_stats(list(det_eval_history), mastery_diff=current_difficulty)
        print(f"  [Det] eps={det_stats['episodes']} harvest_mg={det_stats['median_harvested_mg']:.1f} "
              f"p25={det_stats['p25_harvested_mg']:.1f} time_avg_od={det_stats['median_time_avg_od']:.4f} "
              f"crash={det_stats['crash_rate']*100:.1f}%")

        target = ADVANCE_TARGETS.get(current_difficulty)
        criteria_passed = False
        det_criteria_passed = False
        if target is not None and stats["episodes"] >= MASTERY_MIN_EPISODES:
            criteria_passed = (
                stats["median_harvested_mg"] >= target["min_median_harvested_mg"]
                and stats["p25_harvested_mg"] >= target["min_p25_harvested_mg"]
                and stats["crash_rate"] <= target["max_crash_rate"]
                and stats["median_time_avg_od"] >= target["min_median_time_avg_od"]
            )
        if target is not None and det_stats["episodes"] >= DET_MASTERY_MIN_EPISODES:
            det_criteria_passed = (
                det_stats["median_harvested_mg"] >= target["min_median_harvested_mg"]
                and det_stats["p25_harvested_mg"] >= target["min_p25_harvested_mg"]
                and det_stats["crash_rate"] <= target["max_crash_rate"]
                and det_stats["median_time_avg_od"] >= target["min_median_time_avg_od"]
            )
        criteria_passed = criteria_passed and det_criteria_passed

        next_difficulty = current_difficulty
        if stats["episodes"] >= MASTERY_MIN_EPISODES:
            if criteria_passed:
                mastery_streak += 1
                demotion_streak = 0
            else:
                mastery_streak = 0
            if mastery_streak >= MASTERY_REQUIRED_STREAK:
                next_difficulty = min(2, current_difficulty + 1)
                mastery_streak = 0

            if current_difficulty > 0 and stats["crash_rate"] >= 0.35:
                demotion_streak += 1
            else:
                demotion_streak = 0
            if demotion_streak >= 2:
                next_difficulty = max(0, current_difficulty - 1)
                mastery_streak = 0
                demotion_streak = 0

        if next_difficulty != current_difficulty:
            direction = "ADVANCED" if next_difficulty > current_difficulty else "DEMOTED"
            print(
                f"  Curriculum {direction}: D{current_difficulty} -> D{next_difficulty} "
                f"| chunk_eps={episodes_this_chunk} harvest_mg={stats['median_harvested_mg']:.1f} "
                f"p25={stats['p25_harvested_mg']:.1f} time_avg_od={stats['median_time_avg_od']:.4f} "
                f"crash={stats['crash_rate']:.2%}"
            )
        else:
            print(
                f"  Curriculum hold D{current_difficulty} | chunk_eps={episodes_this_chunk} "
                f"eps={stats['episodes']} harvest_mg={stats['median_harvested_mg']:.1f} "
                f"p25={stats['p25_harvested_mg']:.1f} time_avg_od={stats['median_time_avg_od']:.4f} "
                f"crash={stats['crash_rate']:.2%} adv={mastery_streak}/{MASTERY_REQUIRED_STREAK} "
                f"dem={demotion_streak}/2"
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
    # Fix (v27): NOT YET UPDATED for the macro-timestep/ensemble/two-hot rewrite below this
    # function still uses the pre-Fix#27 4D action, raw-step transitions, and the removed
    # q1/q2 attributes — it would fail with a confusing AttributeError deep in agent.load()/
    # update() rather than a clear one here. Failing loudly at the entry point instead of
    # leaving it silently inconsistent with train_td_mpc2().
    raise NotImplementedError(
        "finetune_td_mpc2() predates Fix #27 (3D action space, macro-timestep world model, "
        "5-critic ensemble, two-hot regression, project curriculum gate) and has not been "
        "updated to match. It will not load a checkpoint saved by the current train_td_mpc2() "
        "correctly. Update this function the same way train_td_mpc2() was updated before use."
    )
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

    raw_env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=3000, difficulty=2)
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
    parser.add_argument("--steps", type=int, default=None,
                        help="Total training step budget (default: 8,000,000, matching PPO).")
    args = parser.parse_args()

    if args.finetune is not None:
        finetune_td_mpc2(extra_steps=args.finetune, use_privileged_distill=args.priv_distill)
    else:
        train_td_mpc2(resume=args.resume, use_privileged_distill=args.priv_distill,
                      total_steps=args.steps)
