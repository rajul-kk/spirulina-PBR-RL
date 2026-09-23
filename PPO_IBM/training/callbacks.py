"""SB3 callbacks used by the recurrent PPO curriculum trainer."""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--training-callbacks-py-3)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
from collections import defaultdict, deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from curriculum_starts import STITCH_POP_THRESHOLD


class TQDMActionCallback(BaseCallback):
    """Appends all 3 raw actuator outputs (Stir, Light, Harvest) and ...
    (full rationale: docs/decision_history.md#--training-callbacks-py-21)"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_ods = []
        self._last_od = 0.0
        self.train_diff = None
        self.mastery_diff = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            self._last_od = max(self._last_od, info.get("od", 0.0))
            if info.get("episode"):
                self._ep_ods.append(self._last_od)
                self._last_od = 0.0
        mean_od = np.mean(self._ep_ods[-10:]) if self._ep_ods else 0.0

        # self.locals is a dict and "callback" is SB3's CallbackList; the progress bar lives on
        # the ProgressBarCallback inside it. (This used hasattr() on the dict, so the postfix
        # never showed.)
        cb = self.locals.get("callback")
        pbar = next((c.pbar for c in getattr(cb, "callbacks", [cb]) if getattr(c, "pbar", None) is not None), None)
        actions = self.locals.get("actions")
        if pbar is not None and actions is not None and len(actions) > 0:
            act = actions[0]
            postfix = {"OD": f"{mean_od:.4f}"}
            if self.train_diff is not None:
                postfix["Diff"] = f"D{self.train_diff}"
            if self.mastery_diff is not None:
                postfix["Mastery"] = f"D{self.mastery_diff}"
            if len(act) >= 2:
                postfix.update({"Stir": f"{act[0]:.2f}", "Lt": f"{act[1]:.2f}"})
            pbar.set_postfix(postfix, refresh=False)
        return True


class EntropyLoggingCallback(BaseCallback):
    """Logs current entropy coefficient into SB3 train metrics each rollout."""
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.model.logger.record("train/entropy_coef", float(self.model.ent_coef))


class PopulationStitchCallback(BaseCallback):
    """Implements Population-Seeded Batch Stitching for Stable-Baselines3.
    (full rationale: docs/decision_history.md#--training-callbacks-py-68)"""
    def __init__(self, controller,
                 pop_threshold: int = STITCH_POP_THRESHOLD, difficulty_min: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.controller = controller
        self.pop_threshold  = pop_threshold
        self.difficulty_min = difficulty_min

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])

        for done, info in zip(dones, infos):
            if not done:
                continue
            self.controller.completed_episodes += 1
            # The vec env has already auto-reset the raw env, so its state is the NEXT
            # episode's; the finished culture arrives as info["terminal_population"],
            # attached by CurriculumStartWrapper.step() before the reset.
            snap = info.get("terminal_population")
            num_active = int(info.get("pop", 0))
            if (snap is not None and num_active >= self.pop_threshold
                    and int(info.get("episode_train_diff", 0)) >= self.difficulty_min):
                self.controller.saved_state = snap
                if self.verbose:
                    print(f"[Stitch] Saved state with {num_active:,} cells (OD={info.get('od', 0.0):.4f})")
        return True



class EpisodeMetricsCallback(BaseCallback):
    """Collect episode-end metrics used for adaptive curriculum decisions.
    (full rationale: docs/decision_history.md#--training-callbacks-py-121)"""
    def __init__(self, window_size: int = 40, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.history_by_diff = defaultdict(lambda: deque(maxlen=window_size))
        self.episode_metrics = []  # this chunk's episodes only (std-control pacing, logging)

    def start_new_chunk(self):
        self.episode_metrics = []

    def history_for_difficulty(self, difficulty: int):
        return list(self.history_by_diff[difficulty])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for idx, done in enumerate(dones):
            if not done:
                continue
            info = infos[idx] if idx < len(infos) else {}
            ep_info = info.get("episode", {})
            reward = float(ep_info.get("r", 0.0))
            ep_len = int(ep_info.get("l", 0))
            reward_per_step = reward / max(ep_len, 1)
            harvested_mg = float(info.get("cumulative_harvested_mg", 0.0))
            time_avg_od = float(info.get("time_avg_od", 0.0))

            # Read the env's own verdict, not raw_env.num_active: the vec env auto-resets
            # before callbacks run, so that was the NEXT episode's starting population
            # (never < 10), which made every stochastic crash rate 0%.
            crashed = not info.get("TimeLimit.truncated", False)
            # episode_train_diff is injected by CurriculumStartWrapper.step() on done,
            # (full rationale: docs/decision_history.md#--training-callbacks-py-169)
            ep_train_diff = int(info.get("episode_train_diff", -1))
            record = {
                "reward": reward_per_step,
                "harvested_mg": harvested_mg,
                "time_avg_od": time_avg_od,
                "episode_duration_h": ep_len * 0.02,
                "crashed": crashed,
                "start_mode": info.get("start_mode", "low"),
                "train_diff": ep_train_diff,
            }
            self.episode_metrics.append(record)
            self.history_by_diff[ep_train_diff].append(record)
        return True
