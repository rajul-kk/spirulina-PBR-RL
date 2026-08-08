"""Entropy coefficient scheduling and policy-std control for the curriculum trainer."""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# This module lives in a subdirectory but imports project modules flatly (e.g.
# `from env_utils import ...`) and expects `environments/` importable. Add the repo root,
# training/ and environments/ to sys.path so those imports resolve regardless of which
# directory this file sits in. Run all scripts FROM THE REPO ROOT: relative paths like
# "model_data/..." are resolved against the working directory, not against __file__.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import numpy as np
import torch

ENTROPY_INIT = 0.02
ENTROPY_DECAY = 0.985
ENTROPY_MIN = 0.003
ENTROPY_MAX = 0.20
# Fix #12 (v16): lowered 0.20 -> 0.08. This constant is the *floor* below which the std-band
# controller pushes ent_coef UP (recurrent_ppo.py: `latest_std < STD_BAND_LOW` -> multiplier +=
# ENTROPY_ADJUST_UP), so it sets a hard lower bound on how tightly the policy is ever allowed to
# converge. At 0.20 that bound was actively preventing convergence, and it directly contradicted
# this file's own note below describing "healthy 0.03-0.08" std — the controller was defending a
# std 2.5-6x above what the same comment calls healthy.
#
# Measured evidence (v16 pre-work, D1): a fixed-action physics sweep at BOTH the reference
# operating point (stir=80/light=1000) and the v15 policy's own measured one (stir=60/light=900)
# put the reward optimum at harvest frac 0.18-0.20 (mean reward ~1117), and that optimum also
# clears the D1 curriculum gate with margin on every criterion. The v15 trained policy instead
# sat at frac 0.30-0.44 across all seeds and both archives, earning only ~842 (frac 0.30) to
# ~646 (frac 0.35) — i.e. it left 250-470 reward per episode unclaimed, so it was NOT at its own
# reward optimum and this is not a reward-shape problem (Fix #10's peaked reward_od is working as
# designed: its optimum sits squarely inside the gate-passing window).
# The mechanism: the 6.5M archive's decoded harvest std was 0.05-0.09, and the raw->frac map has
# slope 0.25 (raw [-1,1] -> frac [0,0.5]), so decoded 0.05 == raw std 0.20 — pinned *exactly* at
# STD_BAND_LOW. The controller was holding the policy at its own floor, making it unable to hold
# any steady harvest fraction, and the sweep shows time_avg_od is monotonically decreasing in
# frac, so drifting frac directly explains the decaying deterministic time_avg_od that blocked
# every D1->D2 attempt.
# 0.08 is the top of this file's own stated healthy range: genuine std collapse is still caught,
# but convergence into 0.08-0.65 is now permitted instead of fought.
STD_BAND_LOW = 0.08
STD_BAND_HIGH = 0.65   # raised from 0.45: std 0.5-0.6 is healthy early-D0 exploration
STD_HARD_CAP = STD_BAND_HIGH
STD_CONTROL_EVERY_EPISODES = 10
ENTROPY_ADJUST_UP = 0.16
ENTROPY_ADJUST_DOWN = 0.25  # proportional decay factor (mult *= this when std > band_high)
ENTROPY_RELAX_STEP = 0.05
ENTROPY_MULT_MIN = 0.20
ENTROPY_MULT_MAX = 1.80
ENTROPY_PLATEAU_CAP = 1.3  # plateau kick hard ceiling
# Lowered from 3.6x -> 2.0x: repeated plateau kicks compounding up to 3.0x caused the
# policy's action distribution to become near-random (std 0.5-0.9 vs healthy 0.03-0.08),
# which then trained the network on garbage rollouts and produced a real, non-recovering
# reward regression (-75 peak -> -260) even after the multiplier decayed back down.
# Lowered again 2.0x -> 1.3x: a full training run at 2.0x reproduced the same failure
# signature (a live "Std hard-cap applied: 0.714 -> 0.484 (cap=0.65)" correction logged
# right after plateau kicks pushed the multiplier to 2.0x) — the plateau mechanism's
# aggressive upward kick was directly fighting the std-band feedback loop below (which
# already has its own escalation path via ENTROPY_ADJUST_UP), and there's up to a full
# 100k-step chunk of lag between a kick and the next std correction. The policy
# demonstrably could already clear its curriculum gate under high entropy (twice, in
# that run) but couldn't sustain it — consistent with exploration noise, not insufficient
# exploration, being the limiting factor once training has progressed this far.
STD_LOW_PUSH_MIN_ENT_COEF = ENTROPY_MIN * 1.6

# ── Fix #22 (v24): late-training policy-std annealing ────────────────────────────────────
# THE PROBLEM. Every run's DETERMINISTIC performance is far worse than its stochastic
# performance — v16b det 20-48mg vs stoch 85-212mg; v22 det 39.7mg/od 0.0086 vs stoch
# 211mg/od 0.0209. The curriculum gate and any real deployment use the deterministic (mean)
# policy; PPO optimises the stochastic one. So the agent is graded on a criterion it is
# never trained on, and the gate reads as an obstacle rather than a filter.
#
# WHY THE MEAN LOSES TO ITS OWN SAMPLES. The harvest action is clipped at 0. When the
# policy's mean sits near that floor, samples can only deviate UPWARD, so the sampled
# policy harvests substantially while the mean harvests almost nothing — E[f(x)] != f(E[x]),
# asymmetrically, because of the boundary. This predicts the gap shrinks when the mean sits
# in the interior, and that is observed: v21's mean harvest fraction was 0.16-0.18 (off the
# floor) and it had both the best deterministic numbers and the smallest gap of any run.
#
# WHY IT NEVER RESOLVES ITSELF. `train/std` sat at ~0.50-0.54 in every single run
# (v23: 0.543 at chunk 74). The entropy schedule actively sustains it: with weak advantage
# gradients the entropy bonus dominates and std equilibrates high. Fix #12a's STD_BAND_LOW
# reduction was inert precisely because std never descended far enough to touch it.
#
# THE FIX. Rather than hoping entropy tuning lowers std, explicitly anneal a HARD CAP on it
# over the back half of training, so mean and samples converge and the two objectives stop
# diverging. Deliberately schedule-based, not reactive: the std-band controller is reactive
# and has demonstrably failed to bring std down for 13 runs.
# STD_ANNEAL_FINAL is kept comfortably ABOVE STD_BAND_LOW (0.08) so the two controllers
# cannot fight — a cap below the band floor would have the annealer clamp std down while the
# band controller pushes entropy up to raise it, the same conflict Fix #12b had to bound.
STD_ANNEAL_START_FRAC = 0.40   # progress fraction at which annealing begins
STD_ANNEAL_END_FRAC = 0.85     # fully annealed by here, leaving a stable tail
STD_ANNEAL_FINAL = 0.12        # final cap (vs STD_BAND_LOW=0.08, so 0.04 of margin)


def annealed_std_cap(progress: float) -> float:
    """Hard cap on actor std as a function of overall training progress in [0, 1].

    Flat at STD_HARD_CAP until STD_ANNEAL_START_FRAC, then linear down to
    STD_ANNEAL_FINAL by STD_ANNEAL_END_FRAC. Early training keeps full exploration; late
    training forces the mean policy to become the policy that is actually evaluated.
    """
    p = float(np.clip(progress, 0.0, 1.0))
    if p <= STD_ANNEAL_START_FRAC:
        return STD_HARD_CAP
    if p >= STD_ANNEAL_END_FRAC:
        return STD_ANNEAL_FINAL
    t = (p - STD_ANNEAL_START_FRAC) / (STD_ANNEAL_END_FRAC - STD_ANNEAL_START_FRAC)
    return float(STD_HARD_CAP + (STD_ANNEAL_FINAL - STD_HARD_CAP) * t)


def entropy_decay_value(chunk_number: int) -> float:
    """Simple exponential entropy decay applied per training chunk."""
    idx = max(0, int(chunk_number) - 1)
    return float(max(ENTROPY_MIN, ENTROPY_INIT * (ENTROPY_DECAY ** idx)))


def entropy_hybrid_value(chunk_number: int, entropy_multiplier: float) -> float:
    """Decay baseline with a bounded multiplier for low/high std correction."""
    base = entropy_decay_value(chunk_number)
    return float(np.clip(base * entropy_multiplier, ENTROPY_MIN, ENTROPY_MAX))


def clamp_policy_std(model, std_cap: float):
    """Upper-bound actor std directly via log_std when exploration runs away."""
    if not hasattr(model, "policy") or not hasattr(model.policy, "log_std"):
        return None

    max_log_std = float(np.log(max(std_cap, 1e-8)))
    with torch.no_grad():
        log_std_param = model.policy.log_std
        std_before = float(torch.exp(log_std_param).mean().item())
        log_std_param.data.clamp_(max=max_log_std)
        std_after = float(torch.exp(log_std_param).mean().item())
    return std_before, std_after
