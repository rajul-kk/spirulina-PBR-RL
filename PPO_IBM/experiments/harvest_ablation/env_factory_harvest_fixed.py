"""env_factory_harvest_fixed.py — harvest-ablation variant of training/env_factory.py.
Same wrapper stack, with HarvestFixedWrapper inserted innermost so the fixed harvest
value is seen by every downstream wrapper, not just the raw env physics."""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from stable_baselines3.common.monitor import Monitor

from genetic_env import GeneticPhotobioreactorEnv
from wrappers import ActionSmoothnessWrapper, ACTION_SMOOTH_WRAPPER_PENALTY, HarvestFixedWrapper
from curriculum_schedule import CurriculumStartController, CurriculumStartWrapper

# frac=0.15, best sustainable constant per bc_pretrain.py and env_diagnosis (see
# experiments/harvest_ablation/README.md). raw = interp(0.15, [0, F_MAX=0.5], [-1,1]).
FIXED_HARVEST_RAW = -0.4


def make_env(difficulty: int, controller: CurriculumStartController, initial_cells: int = 300):
    def _init():
        env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=initial_cells)
        env = HarvestFixedWrapper(env, FIXED_HARVEST_RAW)
        env = ActionSmoothnessWrapper(env, penalty_coef=ACTION_SMOOTH_WRAPPER_PENALTY)
        env = CurriculumStartWrapper(env, controller)
        return Monitor(env)
    return _init
