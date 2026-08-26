"""SB3 callbacks used by the recurrent PPO curriculum trainer."""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--training-callbacks-py-3)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import copy
from collections import defaultdict, deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from env_utils import unwrap_raw_env


class TQDMActionCallback(BaseCallback):
    """
    Appends all 3 raw actuator outputs (Stir, Light, Harvest) and
    rolling mean OD to the TQDM progress bar on every env step.
    """
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

        if hasattr(self.locals, "callback") and hasattr(self.locals["callback"], "pbar"):
            pbar = self.locals["callback"].pbar
            if pbar is not None:
                actions = self.locals.get("actions")
                if actions is not None and len(actions) > 0:
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
    """
    Implements Population-Seeded Batch Stitching for Stable-Baselines3.

    On episode end: if num_active > pop_threshold, save the full physical state.
    Reset-time start selection is handled by CurriculumStartWrapper.
    """
    def __init__(self, controller,
                 pop_threshold: int = 15_000, difficulty_min: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.controller = controller
        self.pop_threshold  = pop_threshold
        self.difficulty_min = difficulty_min

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])

        for done, info in zip(dones, infos):
            raw_env = unwrap_raw_env(self.training_env)

            # --- On episode END: save state if population was high ---
            if done:
                self.controller.completed_episodes += 1
                num_active = getattr(raw_env, 'num_active', 0)
                if num_active >= self.pop_threshold:
                    self.controller.saved_state = {
                        'cells_mass':   copy.deepcopy(raw_env.cells_mass),
                        'cells_quota':  copy.deepcopy(raw_env.cells_quota),
                        'cells_z':      copy.deepcopy(raw_env.cells_z),
                        'clump_mass':   copy.deepcopy(raw_env.clump_mass),
                        'pigment':      raw_env.pigment,
                        'num_active':   raw_env.num_active,
                        'active_mask':  copy.deepcopy(raw_env.active_mask),
                        'ext_nutrients':raw_env.ext_nutrients,
                        'p_pool':       raw_env.p_pool,
                        'ph':           raw_env.ph,
                        'do2':          raw_env.do2,
                        'do2_s':        getattr(raw_env, 'do2_s', raw_env.do2),
                        'do2_b':        getattr(raw_env, 'do2_b', raw_env.do2),
                        'co2_s':        getattr(raw_env, 'co2_s', 2.0),
                        'co2_b':        getattr(raw_env, 'co2_b', 2.0),
                        'salt':         raw_env.salt,
                    }
                    # Also save cells_x if the env is 2D (genetic/total)
                    if hasattr(raw_env, 'cells_x'):
                        self.controller.saved_state['cells_x'] = copy.deepcopy(raw_env.cells_x)
                    if self.verbose:
                        print(f"[Stitch] Saved state with {num_active:,} cells "
                              f"(OD={getattr(raw_env, 'od', 0):.4f})")
        return True


class EpisodeMetricsCallback(BaseCallback):
    """Collect episode-end metrics used for adaptive curriculum decisions.

    Maintains a persistent, per-difficulty rolling window (deque, maxlen=window_size)
    that survives across chunk boundaries, instead of a flat list that used to be
    discarded (a fresh EpisodeMetricsCallback instantiated) every 100k-step chunk.
    That previously meant curriculum advancement/demotion decisions were made on
    whatever ~14 episodes happened to land in the current chunk — a sample small and
    narrow enough that a "lucky" chunk (biased toward larger, easier initial
    populations) could pass a gate that didn't hold up on a broader held-out sample
    (see held_out_sweep.py). This instance should be constructed once and reused
    across the whole training run's chunk loop; call start_new_chunk() at each chunk
    boundary to reset only the per-chunk episode counter (still needed for entropy
    std-control pacing), not the rolling history.
    """
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
        raw_env = unwrap_raw_env(self.training_env)

        for idx, done in enumerate(dones):
            if not done:
                continue
            info = infos[idx] if idx < len(infos) else {}
            ep_info = info.get("episode", {})
            reward = float(ep_info.get("r", 0.0))
            ep_len = int(ep_info.get("l", 0))
            reward_per_step = reward / max(ep_len, 1)
            final_pop = int(getattr(raw_env, "num_active", 0))
            harvested_mg = float(info.get("cumulative_harvested_mg", 0.0))
            time_avg_od = float(info.get("time_avg_od", 0.0))

            crashed = final_pop < 10
            # episode_train_diff is injected by CurriculumStartWrapper.step() on done,
            # (full rationale: docs/decision_history.md#--training-callbacks-py-169)
            ep_train_diff = int(info.get("episode_train_diff", -1))
            record = {
                "reward": reward_per_step,
                "harvested_mg": harvested_mg,
                "time_avg_od": time_avg_od,
                "episode_duration_h": ep_len * 0.02,
                "crashed": crashed,
                "start_mode": info.get("start_mode", getattr(raw_env, "episode_start_mode", "low")),
                "train_diff": ep_train_diff,
            }
            self.episode_metrics.append(record)
            self.history_by_diff[ep_train_diff].append(record)
        return True
