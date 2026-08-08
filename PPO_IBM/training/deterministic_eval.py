"""
deterministic_eval.py — lightweight, read-only deterministic evaluation episode used by the
curriculum training loop (recurrent_ppo.py) to gate advancement, in addition to the existing
stochastic-rollout gate.

Why this exists: EpisodeMetricsCallback records episodes generated during model.learn(),
which always uses stochastic action sampling (the entropy term's whole purpose). A policy
whose deterministic (mean) action has collapsed to a degenerate strategy (e.g. never
harvesting) can still look like it "harvests fine" in the stochastic rollouts purely from
exploration noise around that mean occasionally crossing into a nonzero action — inflating
the live curriculum gate without reflecting what the actually-deployed (deterministic)
policy does. held_out_sweep.py and test_actions.py both catch this because they use
deterministic=True, but neither runs during training. This module brings that same
deterministic evaluation into the training loop itself, cheaply (a handful of episodes per
chunk), so a policy that only "looks like" it works under exploration noise can no longer
advance or be declared mastered.

Modeled directly on held_out_sweep.py's run_episode — same env construction and step loop —
but takes a normalization snapshot (obs_rms) instead of loading one from disk, since this
runs against the live, still-training model rather than a saved checkpoint.
"""

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
import copy

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from genetic_env import GeneticPhotobioreactorEnv
from curriculum_schedule import _sample_init_cells


def run_deterministic_eval_episode(model, obs_rms, difficulty, seed=None):
    """Run one full deterministic episode against a fresh env, isolated from the live
    training vec_env (a separate VecNormalize copy, training=False) so this is guaranteed
    read-only — it cannot perturb training's running normalization stats or LSTM state.
    """
    if seed is not None:
        np.random.seed(seed)
    init_cells = _sample_init_cells("random", difficulty)

    def _make():
        env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
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
            # DummyVecEnv auto-resets on done, zeroing the raw env's running totals —
            # the pre-reset values are preserved in the info dict, so read from there.
            harvested_mg = float(info[0].get("cumulative_harvested_mg", 0.0))
            time_avg_od = float(info[0].get("time_avg_od", 0.0))

    return {
        "harvested_mg": harvested_mg,
        "time_avg_od": time_avg_od,
        "crashed": step < max_steps,
        "start_mode": "low",
        "train_diff": difficulty,
        "reward": 0.0,  # unused by _compute_curriculum_stats' gate criteria, kept for shape parity
    }
