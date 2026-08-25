"""
deterministic_eval_harvest_fixed.py — harvest-ablation variant of training/deterministic_eval.py.

Identical to the original except the env is wrapped with HarvestFixedWrapper before
Monitor/VecNormalize, so the deterministic-eval side of the dual gate sees the same
fixed-harvest environment training does. Without this, det-eval would score the agent's
UNUSED harvest output against the real, controllable harvest dimension — a mismatch with
what was actually trained, and a false read on the ablation.
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import copy

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from genetic_env import GeneticPhotobioreactorEnv
from curriculum_schedule import _sample_init_cells
from wrappers import HarvestFixedWrapper

FIXED_HARVEST_RAW = -0.4  # frac=0.15 — see experiments/harvest_ablation/README.md


def run_deterministic_eval_episode(model, obs_rms, difficulty, seed=None):
    if seed is not None:
        np.random.seed(seed)
    init_cells = _sample_init_cells("random", difficulty)

    def _make():
        env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
        env = HarvestFixedWrapper(env, FIXED_HARVEST_RAW)
        return Monitor(env)

    base = DummyVecEnv([_make])
    vec = VecNormalize(base, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec.obs_rms = copy.deepcopy(obs_rms)
    vec.training = False

    obs = vec.reset()
    max_steps = vec.venv.envs[0].env.max_steps
    lstm_states = None
    ep_starts = np.ones((1,), dtype=bool)
    done = False
    step = 0
    harvested_mg = 0.0
    time_avg_od = 0.0
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
        ep_starts = np.zeros((1,), dtype=bool)
        obs, reward, done_vec, info = vec.step(action)
        done = bool(done_vec[0])
        step += 1
        if done:
            harvested_mg = float(info[0].get("cumulative_harvested_mg", 0.0))
            time_avg_od = float(info[0].get("time_avg_od", 0.0))

    return {
        "harvested_mg": harvested_mg,
        "time_avg_od": time_avg_od,
        "crashed": step < max_steps,
        "start_mode": "low",
        "train_diff": difficulty,
        "reward": 0.0,
    }
