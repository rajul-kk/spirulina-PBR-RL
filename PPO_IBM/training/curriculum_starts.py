
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--training-curriculum_starts-py-2)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import copy
from typing import Dict, Iterable, List, Optional

import numpy as np


START_DISTRIBUTION = {
    0: {"low": 0.85, "mid": 0.10, "stitched": 0.05},
    1: {"low": 0.35, "mid": 0.35, "high": 0.10, "stitched": 0.20},
    2: {"low": 0.20, "mid": 0.20, "high": 0.15, "stitched": 0.45},
}

STITCH_MIN_EPISODES = {
    0: 20,
    1: 0,
    2: 0,
}

MAX_STITCHED_SHARE_FOR_MASTERY = 0.30


def _log_uniform_int(low: int, high: int, rng) -> int:
    return int(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_initial_cells(difficulty: int, bucket: str, rng=np.random) -> int:
    if bucket == "high":
        return _log_uniform_int(2000, 5000, rng)

    if bucket == "mid":
        return _log_uniform_int(600, 1500, rng)

    if difficulty == 2 and rng.rand() < 0.10:
        return int(rng.uniform(30, 80))

    return _log_uniform_int(100, 400, rng)


def choose_episode_start(
    difficulty: int,
    saved_state_available: bool,
    completed_episodes: int,
    rng=np.random,
) -> Dict[str, Optional[int]]:
    difficulty = int(np.clip(difficulty, 0, 2))
    weights = dict(START_DISTRIBUTION[difficulty])
    if (not saved_state_available) or (completed_episodes < STITCH_MIN_EPISODES[difficulty]):
        weights.pop("stitched", None)

    modes = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=np.float64)
    probs /= probs.sum()
    mode = str(rng.choice(modes, p=probs))

    if mode == "stitched":
        return {"mode": mode, "initial_cells": None}

    return {"mode": mode, "initial_cells": sample_initial_cells(difficulty, mode, rng=rng)}


def apply_saved_population(raw_env, saved_state: Dict[str, object], start_mode: str = "stitched") -> None:
    # Guard: discard states saved under a different max_cells (e.g. after super-agent rescaling).
    # Mismatched array sizes would silently corrupt mass/mask operations.
    saved_size = len(saved_state.get("cells_mass", []))
    if saved_size != raw_env.max_cells:
        return
    raw_env.cells_mass = copy.deepcopy(saved_state["cells_mass"])
    raw_env.cells_quota = copy.deepcopy(saved_state["cells_quota"])
    raw_env.cells_z = copy.deepcopy(saved_state["cells_z"])
    raw_env.clump_mass = copy.deepcopy(saved_state["clump_mass"])
    raw_env.pigment = saved_state["pigment"]
    raw_env.num_active = saved_state["num_active"]
    raw_env.active_mask = copy.deepcopy(saved_state["active_mask"])
    raw_env.ext_nutrients = saved_state["ext_nutrients"]
    raw_env.p_pool        = float(saved_state.get("p_pool", 80.0))
    raw_env.do2_s         = float(saved_state.get("do2_s", saved_state.get("do2", 7.0)))
    raw_env.do2_b         = float(saved_state.get("do2_b", saved_state.get("do2", 7.0)))
    raw_env.co2_s         = float(saved_state.get("co2_s", 2.0))
    raw_env.co2_b         = float(saved_state.get("co2_b", 2.0))
    # Keep stitched starts from inheriting legacy low-pH snapshots.
    # For alkaline media envs, enforce at least the configured equilibrium pH.
    saved_ph = float(saved_state["ph"])
    ph_floor = float(getattr(raw_env, "buffer_equilibrium_ph", saved_ph))
    raw_env.ph = max(saved_ph, ph_floor)
    if hasattr(raw_env, "_ph_obs_ema"):
        raw_env._ph_obs_ema = raw_env.ph
    raw_env.do2 = saved_state["do2"]
    raw_env.salt = saved_state["salt"]
    if "cells_x" in saved_state and hasattr(raw_env, "cells_x"):
        raw_env.cells_x = copy.deepcopy(saved_state["cells_x"])
    raw_env.dosing_integral = 0.0    # PID dosing history unknown for stitched starts
    raw_env.harvest_integral = 0.0   # harvest pump counter unknown for stitched starts
    raw_env.current_harvest_rate = 0.0
    raw_env.cumulative_harvested_mg = 0.0  # curriculum metric — episode-scoped, must reset
    raw_env.od_sum_back_half = 0.0
    raw_env.od_count_back_half = 0
    raw_env.I_surface = 0.0          # reset BH1750 source; will update on first step
    raw_env.episode_start_mode = start_mode


def cap_stitched_metrics(
    episode_metrics: Iterable[Dict[str, object]],
    max_stitched_share: float = MAX_STITCHED_SHARE_FOR_MASTERY,
) -> List[Dict[str, object]]:
    metrics = list(episode_metrics)
    if not metrics:
        return []

    base = [metric for metric in metrics if metric.get("start_mode") != "stitched"]
    stitched = [metric for metric in metrics if metric.get("start_mode") == "stitched"]

    if not base or not stitched:
        return metrics

    max_stitched = int(np.floor((max_stitched_share * len(base)) / max(1e-8, 1.0 - max_stitched_share)))
    if max_stitched <= 0:
        return base

    return base + stitched[-max_stitched:]


def mastery_metrics_view(
    episode_metrics: Iterable[Dict[str, object]],
    max_stitched_share: float = MAX_STITCHED_SHARE_FOR_MASTERY,
) -> List[Dict[str, object]]:
    """Return metrics used for curriculum pass/fail.

    Policy: exclude stitched episodes when non-stitched episodes exist.
    If a window has only stitched episodes, fall back to capped mixed view.
    """
    metrics = list(episode_metrics)
    if not metrics:
        return []

    non_stitched = [m for m in metrics if m.get("start_mode") != "stitched"]
    if non_stitched:
        return non_stitched

    return cap_stitched_metrics(metrics, max_stitched_share=max_stitched_share)