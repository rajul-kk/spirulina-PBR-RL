"""Env factory for the recurrent PPO curriculum trainer."""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--training-env_factory-py-3)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
from stable_baselines3.common.monitor import Monitor

from genetic_env import GeneticPhotobioreactorEnv
from wrappers import ActionSmoothnessWrapper, ACTION_SMOOTH_WRAPPER_PENALTY
from curriculum_schedule import CurriculumStartController, CurriculumStartWrapper


def make_env(difficulty: int, controller: CurriculumStartController, initial_cells: int = 300):
    """Factory that creates a single monitored env at the given difficulty."""
    def _init():
        env = GeneticPhotobioreactorEnv(
            max_cells=7_500,    # super-agent: 10× mass/particle → 10× fewer particles, same OD cap
            initial_cells=initial_cells
        )
        env = ActionSmoothnessWrapper(env, penalty_coef=ACTION_SMOOTH_WRAPPER_PENALTY)
        env = CurriculumStartWrapper(env, controller)
        return Monitor(env)
    return _init
