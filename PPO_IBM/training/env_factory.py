"""Env factory for the recurrent PPO curriculum trainer."""

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
