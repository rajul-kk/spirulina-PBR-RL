"""Curriculum difficulty scheduling: mastery/demotion targets, per-episode difficulty
sampling, and the reset-time wrapper that applies the sampled difficulty and start mode.
"""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--training-curriculum_schedule-py-5)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import numpy as np
import gymnasium as gym

from curriculum_starts import apply_saved_population, choose_episode_start, mastery_metrics_view

TOTAL_TRAINING_STEPS = 8_000_000
CHUNK_STEPS = 100_000
# MASTERY_WINDOW: size of the persistent, per-difficulty rolling episode-history buffer
# (full rationale: docs/decision_history.md#--training-curriculum_schedule-py-24)
MASTERY_WINDOW = 40
MASTERY_MIN_EPISODES = 20  # raised from 5 so a decision can't fire off a half-full window
MASTERY_REQUIRED_STREAK = 2

# Deterministic-evaluation gate (see deterministic_eval.py): a policy that only "looks like"
# (full rationale: docs/decision_history.md#--training-curriculum_schedule-py-37)
DET_EVAL_EPISODES_PER_CHUNK = 3   # ~3 full 7200-step deterministic episodes per 100k-step chunk
DET_EVAL_WINDOW = 15              # smaller than MASTERY_WINDOW — deterministic episodes cost
                                   # ~5x more per chunk (3 vs the ~14 stochastic episodes/chunk
                                   # collected "for free" as part of the rollout anyway)
DET_MASTERY_MIN_EPISODES = 9      # 3 chunks' worth before eligible, same ratio as the
                                   # stochastic gate (20 of 40)
DEMOTION_CRASH_RATE = 0.35  # drop back a difficulty if crash rate exceeds this for 2 chunks
DEMOTION_STREAK_REQUIRED = 2
PLATEAU_CHUNKS = 6          # consecutive chunks with mastery_streak==0 before entropy kick
# Fix #12 (v16), second half: bound the NUMBER of plateau kicks per difficulty, not just their
# (full rationale: docs/decision_history.md#--training-curriculum_schedule-py-54)
MAX_PLATEAU_KICKS_PER_DIFFICULTY = 2

# Fix #15 (v18): consecutive chunks failing the DETERMINISTIC gate at D1/D2 before demoting a
# (full rationale: docs/decision_history.md#--training-curriculum_schedule-py-67)
CAPABILITY_DEMOTION_CHUNKS = 12

# Advancement criteria applied to cold-start (non-stitched) episodes only.
# (full rationale: docs/decision_history.md#--training-curriculum_schedule-py-75)
ADVANCE_TARGETS = {
    0: {"min_median_harvested_mg": 30.0, "min_p25_harvested_mg": 15.0, "max_crash_rate": 0.15,
        "min_median_time_avg_od": 0.004},
    1: {"min_median_harvested_mg": 60.0, "min_p25_harvested_mg": 30.0, "max_crash_rate": 0.10,
        "min_median_time_avg_od": 0.008},
    2: {"min_median_harvested_mg": 90.0, "min_p25_harvested_mg": 50.0, "max_crash_rate": 0.08,
        "min_median_time_avg_od": 0.011},
}

MIXING_PROBS = {
    0: ([0], [1.0]),
    1: ([1, 0], [0.8, 0.2]),
    2: ([2, 1, 0], [0.7, 0.2, 0.1]),
}


def _sample_init_cells(init_cells_cfg, difficulty):
    """Sample initial cells per-episode. Adversarial cold starts at D2."""
    if init_cells_cfg == "random":
        if difficulty == 2 and np.random.rand() < 0.1:
            return int(np.random.uniform(30, 80))   # adversarial cold start
        return int(np.exp(np.random.uniform(np.log(100), np.log(400))))
    return init_cells_cfg


def _sample_training_difficulty(current_difficulty: int) -> int:
    diffs, probs = MIXING_PROBS[current_difficulty]
    return int(np.random.choice(diffs, p=probs))


def _compute_curriculum_stats(history, mastery_diff: int = None):
    filtered_history = mastery_metrics_view(history)
    # Filter to on-level episodes only (train_diff == mastery_diff).
    if mastery_diff is not None:
        on_level = [h for h in filtered_history if h.get("train_diff", -1) == mastery_diff]
        if len(on_level) >= 1:
            filtered_history = on_level
    if not filtered_history:
        return {
            "episodes": 0,
            "median_harvested_mg": 0.0,
            "p25_harvested_mg": 0.0,
            "median_time_avg_od": 0.0,
            "crash_rate": 0.0,
            "reward_std": 0.0,
        }
    # Exclude stitched (warm-start) episodes from advancement stats — stitched starts
    # inherit an already-productive culture, making harvested_mg not comparable to cold starts.
    cold_only = [h for h in filtered_history if h.get("start_mode", "low") != "stitched"]
    adv_set = cold_only if len(cold_only) >= 1 else filtered_history

    harvested_values = np.array([h.get("harvested_mg", 0.0) for h in adv_set], dtype=np.float32)
    time_avg_od_values = np.array([h.get("time_avg_od", 0.0) for h in adv_set], dtype=np.float32)
    crash_values   = np.array([1.0 if h["crashed"] else 0.0 for h in adv_set], dtype=np.float32)
    rewards        = np.array([h["reward"] for h in filtered_history], dtype=np.float32)

    p25 = float(np.percentile(harvested_values, 25))
    return {
        "episodes": int(len(adv_set)),
        "median_harvested_mg": float(np.median(harvested_values)),
        "p25_harvested_mg": p25,
        "median_time_avg_od": float(np.median(time_avg_od_values)),
        "crash_rate": float(np.mean(crash_values)),
        "reward_std": float(np.std(rewards)),
    }


class CurriculumStartController:
    """Shared state for reset-time start sampling and saved warm starts."""
    def __init__(self):
        self.saved_state = None
        self.completed_episodes = 0
        self.train_diff = None
        self.mastery_diff = None
        self.log_episode_starts = False


class CurriculumStartWrapper(gym.Wrapper):
    """Samples low, mid, or stitched starts on every reset."""
    def __init__(self, env, controller: CurriculumStartController):
        super().__init__(env)
        self.controller = controller

    def reset(self, **kwargs):
        raw_env = self.unwrapped

        # ── Per-Episode Difficulty Sampling ──────────────────────────────────
        # (full rationale: docs/decision_history.md#--training-curriculum_schedule-py-190)
        mastery = self.controller.mastery_diff if self.controller.mastery_diff is not None else getattr(raw_env, "difficulty", 0)
        episode_diff = _sample_training_difficulty(mastery)
        self.controller.train_diff = episode_diff
        self._episode_train_diff = episode_diff  # stored for step() injection
        if hasattr(raw_env, "set_difficulty"):
            raw_env.set_difficulty(episode_diff)
        else:
            raw_env.difficulty = int(np.clip(episode_diff, 0, 2))
        # ─────────────────────────────────────────────────────────────────────

        difficulty = episode_diff  # use sampled diff for start-mode selection
        start_cfg = choose_episode_start(
            difficulty,
            saved_state_available=self.controller.saved_state is not None,
            completed_episodes=self.controller.completed_episodes,
        )
        if start_cfg["initial_cells"] is not None:
            raw_env.initial_cells = int(start_cfg["initial_cells"])
        raw_env.episode_start_mode = start_cfg["mode"]

        obs, info = self.env.reset(**kwargs)

        if start_cfg["mode"] == "stitched" and self.controller.saved_state is not None:
            apply_saved_population(raw_env, self.controller.saved_state)
            obs = raw_env._get_obs()

        raw_env.episode_init_cells = int(getattr(raw_env, "num_active", raw_env.initial_cells))

        if self.controller.log_episode_starts:
            ep_num = int(self.controller.completed_episodes) + 1
            print(
                f"[EpisodeStart] ep={ep_num} train_diff=D{episode_diff} "
                f"mastery_diff=D{mastery} start={start_cfg['mode']} "
                f"init={raw_env.episode_init_cells:,}"
            )

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Inject the episode's training difficulty into the info dict on episode end
        # so EpisodeMetricsCallback can record it before the env auto-resets.
        if terminated or truncated:
            info["episode_train_diff"] = getattr(self, "_episode_train_diff",
                                                   getattr(self.unwrapped, "difficulty", 0))
            info["start_mode"] = getattr(self.unwrapped, "episode_start_mode", "low")
        return obs, reward, terminated, truncated, info
