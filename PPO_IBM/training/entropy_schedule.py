"""Entropy coefficient scheduling and policy-std control for the curriculum trainer."""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--training-entropy_schedule-py-3)
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
# (full rationale: docs/decision_history.md#--training-entropy_schedule-py-22)
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
# (full rationale: docs/decision_history.md#--training-entropy_schedule-py-55)
STD_LOW_PUSH_MIN_ENT_COEF = ENTROPY_MIN * 1.6

# ── Fix #22 (v24): late-training policy-std annealing ────────────────────────────────────
# (full rationale: docs/decision_history.md#--training-entropy_schedule-py-70)
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
