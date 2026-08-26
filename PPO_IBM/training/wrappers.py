"""Gym env wrappers used by the recurrent PPO training pipeline."""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--training-wrappers-py-3)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import numpy as np
import gymnasium as gym

ACTION_SMOOTH_WRAPPER_PENALTY = 0.0


class ActionSmoothnessWrapper(gym.Wrapper):
    """Applies an L2 penalty to consecutive actions to prevent actuation chattering."""
    def __init__(self, env, penalty_coef=ACTION_SMOOTH_WRAPPER_PENALTY):
        super().__init__(env)
        self.penalty_coef = penalty_coef
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.penalty_coef > 0.0:
            penalty = self.penalty_coef * np.sum((action - self.prev_action) ** 2)
            reward -= float(penalty)
        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info


class HarvestFixedWrapper(gym.Wrapper):
    """Overrides the harvest action dimension (index 2) with a fixed raw value before
    it reaches the env (experiments/harvest_ablation/). Action space stays 3D — the
    policy still outputs a harvest value, it's just discarded here."""
    def __init__(self, env, fixed_harvest_raw):
        super().__init__(env)
        self.fixed_harvest_raw = float(fixed_harvest_raw)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        action[2] = self.fixed_harvest_raw
        return self.env.step(action)
