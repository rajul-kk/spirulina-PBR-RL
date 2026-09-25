
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

REGRET_BLEND = 0.25
REGRET_MIN_SHARE_OF_BASE = 0.5

# A finished episode's culture is kept for stitched starts when it ends with at least this many
# active cells. Shared by TD3 and PPO; must stay below both trainers' max_cells (7,500).
STITCH_POP_THRESHOLD = 1_100

_POPULATION_ARRAYS = ("cells_mass", "cells_quota", "cells_x", "cells_z", "cells_acclimation",
                      "clump_mass", "active_mask")
_MEDIUM_STATE = ("ext_nutrients", "n_pool", "p_pool", "alkalinity", "dic", "salt",
                 "do2", "do2_s", "do2_b")


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
    bucket_regret: Optional[Dict[str, float]] = None,
    blend: float = REGRET_BLEND,
) -> Dict[str, Optional[int]]:
    difficulty = int(np.clip(difficulty, 0, 2))
    weights = dict(START_DISTRIBUTION[difficulty])
    if (not saved_state_available) or (completed_episodes < STITCH_MIN_EPISODES[difficulty]):
        weights.pop("stitched", None)

    if bucket_regret:
        stitched_w = weights.pop("stitched", 0.0)
        cold_buckets = list(weights.keys())
        base = np.array([weights[b] for b in cold_buckets], dtype=np.float64)
        base /= base.sum()
        regret_vals = np.array([max(bucket_regret.get(b, 0.0), 0.0) for b in cold_buckets], dtype=np.float64)
        if regret_vals.sum() > 1e-9:
            regret_probs = regret_vals / regret_vals.sum()
            blended = (1 - blend) * base + blend * regret_probs
        else:
            blended = base
        floor = REGRET_MIN_SHARE_OF_BASE * base
        blended = np.maximum(blended, floor)
        blended /= blended.sum()
        remaining = 1.0 - stitched_w
        for b, p in zip(cold_buckets, blended):
            weights[b] = float(p * remaining)
        if stitched_w > 0:
            weights["stitched"] = stitched_w

    modes = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=np.float64)
    probs /= probs.sum()
    mode = str(rng.choice(modes, p=probs))

    if mode == "stitched":
        return {"mode": mode, "initial_cells": None}

    return {"mode": mode, "initial_cells": sample_initial_cells(difficulty, mode, rng=rng)}


def snapshot_population(raw_env) -> Dict[str, object]:
    """Copy of a culture's cells and dissolved medium, for a later stitched start."""
    snap = {k: copy.deepcopy(getattr(raw_env, k)) for k in _POPULATION_ARRAYS if hasattr(raw_env, k)}
    snap.update({k: float(getattr(raw_env, k)) for k in _MEDIUM_STATE if hasattr(raw_env, k)})
    snap["num_active"] = int(raw_env.num_active)
    snap["pigment"] = float(raw_env.pigment)
    return snap


def apply_saved_population(raw_env, saved_state: Dict[str, object], start_mode: str = "stitched") -> None:
    """Swap a saved culture into a freshly reset env.

    Whatever the snapshot carries is restored; anything it lacks (older snapshots have no
    acclimation, alkalinity or DIC) keeps the fresh-medium value reset() just set, rather than
    an invented default.
    """
    # Guard: discard states saved under a different max_cells (e.g. after super-agent rescaling).
    # Mismatched array sizes would silently corrupt mass/mask operations.
    saved_size = len(saved_state.get("cells_mass", []))
    if saved_size != raw_env.max_cells:
        return
    for k in _POPULATION_ARRAYS:
        if k in saved_state and saved_state[k] is not None and hasattr(raw_env, k):
            setattr(raw_env, k, copy.deepcopy(saved_state[k]))
    raw_env.num_active = saved_state["num_active"]
    raw_env.pigment = saved_state["pigment"]
    raw_env._aidx_cache = None   # stitched mask differs in position, not only in count
    if "cells_acclimation" not in saved_state:
        # Mid-range of reset()'s initial acclimation draw; leaving these slots at whatever
        # reset() or a dead cell left behind (often 0) put transplanted cells in photo-shock.
        raw_env.cells_acclimation[raw_env.active_mask] = 200.0

    for k in _MEDIUM_STATE:
        if k in saved_state:
            setattr(raw_env, k, float(saved_state[k]))
    for layer in ("do2_s", "do2_b"):   # older snapshots carried only the mixed DO2
        if layer not in saved_state and "do2" in saved_state:
            setattr(raw_env, layer, float(saved_state["do2"]))
    # pH and carbonate species follow from the restored alkalinity and DIC.
    if hasattr(raw_env, "_update_carbonate_speciation"):
        raw_env._update_carbonate_speciation()
    if hasattr(raw_env, "_ph_obs_ema"):
        raw_env._ph_obs_ema = raw_env.ph

    raw_env.dosing_integral = 0.0    # PID dosing history unknown for stitched starts
    raw_env.harvest_integral = 0.0   # harvest pump counter unknown for stitched starts
    raw_env.cumulative_harvested_mg = 0.0  # curriculum metric — episode-scoped, must reset
    raw_env.cumulative_harvested_mg_back_half = 0.0
    raw_env.od_sum_back_half = 0.0
    raw_env.od_count_back_half = 0
    raw_env.I_surface = 0.0          # reset BH1750 source; will update on first step
    raw_env.episode_start_mode = start_mode


def resync_shaping_potential(raw_env) -> None:
    """Re-seed the PBRS potential after apply_saved_population() + _get_obs().

    reset() caches Phi of the fresh culture; the swapped-in population makes that cache stale,
    so the first step would pay gamma*Phi(stitched) - Phi(fresh) -- a spurious reward of up to
    the full Phi range. Must run AFTER _get_obs(), which is what refreshes env.od."""
    if hasattr(raw_env, "_potential"):
        raw_env._phi_prev = raw_env._potential()


def mastery_metrics_view(episode_metrics: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Metrics used for curriculum pass/fail: non-stitched episodes only, unless every
    episode was stitched, in which case all of them.
    (full rationale: docs/decision_history.md#--training-curriculum_starts-py-159)"""
    metrics = list(episode_metrics)
    non_stitched = [m for m in metrics if m.get("start_mode") != "stitched"]
    return non_stitched or metrics
