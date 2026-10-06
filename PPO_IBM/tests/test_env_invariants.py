"""Simulator invariants (environments/genetic_env.py, training/curriculum_starts.py).

Most of these pin the fixed behaviour of bugs that were found and fixed earlier:
  * cells collapsing to x = 0, z = 0 (positions are redrawn uniformly every step now),
  * harvest-dilution refilling with salt 1000 mg/L while reset() used 2500 (one FRESH_MEDIUM now),
  * stitched starts dropping cells_acclimation (transplanted cells went into photo-shock),
  * stale PBRS potential after a stitched swap,
  * strain randomisation not following reset(seed=...).
"""
import numpy as np
import pytest

import curriculum_starts as cs
import genetic_env as ge
from genetic_env import GeneticPhotobioreactorEnv

MAX_CELLS = 7500


def make(difficulty=2, cells=300, seed=7):
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=cells, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    return env, obs


def random_actions(n, seed=0):
    return np.random.RandomState(seed).uniform(-1, 1, (n, 3)).astype(np.float32)


# ------------------------------------------------------------------ rollout invariants -----------
@pytest.mark.parametrize("difficulty", [0, 1, 2])
def test_random_rollout_is_finite_bounded_and_not_collapsed(difficulty):
    env, obs = make(difficulty)
    assert obs.shape == env.observation_space.shape == (6,)
    n_steps = 650                                   # crosses the first harvest event (step 600)
    for a in random_actions(n_steps, seed=difficulty):
        obs, reward, terminated, truncated, info = env.step(a)
        assert obs.shape == (6,)
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if terminated or truncated:
            break
    assert env.step_count >= 600
    m = env.active_mask
    x, z = env.cells_x[m], env.cells_z[m]
    assert len(x) == env.num_active > 50
    # inside the reactor
    assert x.min() >= 0.0 and x.max() <= env.reactor_width
    assert z.min() >= 0.0 and z.max() <= env.reactor_depth
    # spread over the whole cross-section (uniform: std = width/sqrt(12) = 0.289, depth/sqrt(12) = 0.087)
    assert x.std() > 0.2 and z.std() > 0.06
    assert 0.35 < x.mean() < 0.65 and 0.1 < z.mean() < 0.2
    # not piled into the x = 0 / z = 0 corner (the old collapse put every cell there)
    assert np.mean((x < 0.1) & (z < 0.03)) < 0.05
    assert not np.all(x == 0) and not np.all(z == 0)
    # bookkeeping is consistent
    assert env.num_active == int(env.active_mask.sum())
    assert np.all(env.cells_mass[~m] == 0.0) and np.all(env.cells_mass[m] > 0.0)
    assert np.all(np.isfinite(env.cells_acclimation[m])) and np.all(env.cells_acclimation[m] >= 0.0)


def test_observation_stays_inside_declared_space():
    env, obs = make(2)
    for a in random_actions(200, seed=3):
        obs, *_ = env.step(a)
        assert env.observation_space.contains(obs), obs


def test_nan_action_falls_back_to_a_safe_action():
    env, _ = make(1)
    obs, reward, *_ = env.step(np.array([np.nan, 0.0, 0.0], dtype=np.float32))
    assert np.all(np.isfinite(obs)) and np.isfinite(reward)


def test_extended_observation_flag_changes_width(monkeypatch):
    monkeypatch.setattr(GeneticPhotobioreactorEnv, "OBS_EXTENDED", True)
    env, obs = make(1)
    assert obs.shape == env.observation_space.shape == (8,)


# ------------------------------------------------------------------ seeding ----------------------
def rollout(difficulty, seed, n=120):
    env, obs = make(difficulty, seed=seed)
    out = [obs.copy()]
    rewards = []
    for a in random_actions(n, seed=99):
        obs, r, *_ = env.step(a)
        out.append(obs.copy())
        rewards.append(r)
    return np.array(out), np.array(rewards), env


@pytest.mark.parametrize("difficulty", [0, 2])
def test_seeded_episode_prefix_is_deterministic(difficulty):
    o1, r1, e1 = rollout(difficulty, seed=5)
    o2, r2, e2 = rollout(difficulty, seed=5)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(r1, r2)
    assert e1.strain_params == e2.strain_params
    assert e1.num_active == e2.num_active


def test_different_seeds_give_different_strains_and_trajectories():
    o1, _, e1 = rollout(2, seed=5, n=30)
    o2, _, e2 = rollout(2, seed=6, n=30)
    assert e1.strain_params["mu_max"] != e2.strain_params["mu_max"]
    assert not np.array_equal(o1, o2)


def test_reset_seed_reseeds_strain_randomisation():
    """gym's reset(seed) only seeds self.np_random; the env draws from the global np.random, so
    reset() must reseed that itself (regression: held-out sweeps once never seeded the strain)."""
    env, _ = make(2, seed=11)
    p1 = dict(env.strain_params)
    np.random.seed(12345)                      # scramble global state in between
    env.reset(seed=11)
    assert env.strain_params == p1


# ------------------------------------------------------------------ medium: reset vs refill -------
POOLS = ("ext_nutrients", "n_pool", "p_pool", "alkalinity", "dic", "salt")


def test_reset_fills_the_tank_with_fresh_medium():
    fresh = GeneticPhotobioreactorEnv.FRESH_MEDIUM
    assert fresh["salt"] == 2500.0
    env, _ = make(0)
    for pool in POOLS:
        assert getattr(env, pool) == pytest.approx(fresh[pool]), pool
    assert env.do2 == env.do2_s == env.do2_b == fresh["do2"]


@pytest.mark.parametrize("frac", [0.1, 0.3, 0.5])
def test_harvest_refill_uses_the_same_fresh_medium_as_reset(frac):
    fresh = GeneticPhotobioreactorEnv.FRESH_MEDIUM
    env, _ = make(0, cells=400)                     # D0: no pump delivery error, so frac is exact
    for pool in POOLS:
        setattr(env, pool, 0.0)                     # empty tank: whatever refills it shows up directly
    env.do2_s = env.do2_b = 0.0
    env.step_count = env.HARVEST_INTERVAL_STEPS     # the next _apply_harvest is a harvest event
    env._harvest_action_sum, env._harvest_action_count = frac * 10, 10
    mg, is_event = env._apply_harvest()
    assert is_event
    for pool in POOLS:
        assert getattr(env, pool) == pytest.approx(fresh[pool] * frac), pool
    assert env.salt == pytest.approx(2500.0 * frac)       # the bug refilled at 1000 mg/L
    assert env.do2_s == pytest.approx(fresh["do2"] * frac) and env.do2_b == pytest.approx(fresh["do2"] * frac)
    assert env.harvest_integral == pytest.approx(frac * env.volume_L)
    assert mg > 0


def test_repeated_dilution_converges_to_the_reset_medium():
    fresh = GeneticPhotobioreactorEnv.FRESH_MEDIUM
    env, _ = make(0, cells=3000)                    # enough cells to survive 8 half-harvests
    env.salt = 9000.0
    for k in range(1, 9):
        env.step_count = k * env.HARVEST_INTERVAL_STEPS
        env._harvest_action_sum, env._harvest_action_count = 0.5, 1
        env._apply_harvest()
    assert env.salt == pytest.approx(fresh["salt"] + 6500.0 / 2 ** 8)     # exponential relaxation to 2500
    assert abs(env.salt - fresh["salt"]) < 30.0


def test_no_dilution_off_the_harvest_interval():
    env, _ = make(0)
    env.salt = 123.0
    env.step_count = env.HARVEST_INTERVAL_STEPS + 1
    env._harvest_action_sum, env._harvest_action_count = 0.5, 1
    _, is_event = env._apply_harvest()
    assert not is_event and env.salt == 123.0


# ------------------------------------------------------------------ stitched starts ---------------
def culture_after(n_steps=40, difficulty=1):
    env, _ = make(difficulty, cells=600, seed=21)
    for a in random_actions(n_steps, seed=4):
        env.step(a)
    m = env.active_mask
    env.cells_acclimation[m] = np.random.RandomState(0).uniform(500.0, 900.0, int(m.sum()))
    return env


def test_snapshot_carries_acclimation_and_stitch_restores_it():
    src = culture_after()
    snap = cs.snapshot_population(src)
    assert "cells_acclimation" in snap
    dst, _ = make(1, cells=150, seed=3)
    cs.apply_saved_population(dst, snap)
    assert dst.num_active == src.num_active
    np.testing.assert_array_equal(dst.active_mask, src.active_mask)
    np.testing.assert_array_equal(dst.cells_acclimation, src.cells_acclimation)
    assert dst.cells_acclimation[dst.active_mask].min() >= 500.0       # not zeroed / not reset()'s 100-300 draw
    for k in ("salt", "n_pool", "p_pool", "alkalinity", "dic", "do2_s", "do2_b"):
        assert getattr(dst, k) == pytest.approx(getattr(src, k)), k
    assert dst.episode_start_mode == "stitched"
    assert dst.cumulative_harvested_mg == 0.0 and dst.harvest_integral == 0.0


def test_snapshot_is_a_deep_copy():
    src = culture_after()
    snap = cs.snapshot_population(src)
    before = snap["cells_acclimation"].copy()
    src.cells_acclimation[:] = -1.0
    np.testing.assert_array_equal(snap["cells_acclimation"], before)


def test_legacy_snapshot_without_acclimation_gets_midrange_not_zero():
    src = culture_after()
    snap = cs.snapshot_population(src)
    del snap["cells_acclimation"]
    dst, _ = make(1, cells=150, seed=3)
    dst.cells_acclimation[:] = 0.0                                    # what a dead slot looks like
    cs.apply_saved_population(dst, snap)
    assert np.all(dst.cells_acclimation[dst.active_mask] == 200.0)


def test_snapshot_from_a_different_max_cells_is_discarded():
    src = culture_after()
    snap = cs.snapshot_population(src)
    other = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS - 500, initial_cells=100, difficulty=1)
    other.reset(seed=1)
    n = other.num_active
    cs.apply_saved_population(other, snap)
    assert other.num_active == n


def test_stitched_start_resyncs_shaping_potential_and_steps_cleanly():
    src = culture_after()
    snap = cs.snapshot_population(src)
    dst, _ = make(1, cells=150, seed=3)
    cs.apply_saved_population(dst, snap)
    dst._get_obs()                                  # refreshes env.od
    cs.resync_shaping_potential(dst)
    assert dst._phi_prev == pytest.approx(dst._potential())
    obs, reward, *_ = dst.step(np.zeros(3, dtype=np.float32))
    assert np.all(np.isfinite(obs)) and np.isfinite(reward)
    assert abs(reward) < 1.0                        # no spurious gamma*Phi(stitched) - Phi(fresh) jump


def test_stitched_culture_keeps_growing_not_dying():
    """Acclimated cells transplanted into a fresh episode keep their photo-acclimation, so a few
    steps of moderate light do not wipe the culture out."""
    src = culture_after(n_steps=20)
    snap = cs.snapshot_population(src)
    dst, _ = make(1, cells=150, seed=3)
    cs.apply_saved_population(dst, snap)
    n0 = dst.num_active
    for _ in range(40):
        dst.step(np.array([0.0, 0.0, -1.0], dtype=np.float32))
    assert dst.num_active > 0.8 * n0
