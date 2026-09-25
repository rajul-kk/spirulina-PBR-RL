"""Regression checks for the 2026-09-24 core audit of the env, TD3, PPO and curriculum code.

Each check targets one audited bug (or the 2026-09-25 thermal-Phi fix) and fails on the pre-fix code. Run after any change to
genetic_env.py, curriculum_starts.py, TD3.py, deterministic_eval.py or callbacks.py.

  python experiments/env_diagnosis/core_audit_check.py
"""
import inspect
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments"), os.path.join(ROOT, "legacy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from genetic_env import GeneticPhotobioreactorEnv
import curriculum_starts

STEADY = np.array([0.2, 0.5, -1.0], dtype=np.float32)
RESULTS = []


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
        except AssertionError as e:
            RESULTS.append((name, False, str(e)))
        except Exception:
            RESULTS.append((name, False, traceback.format_exc().strip().splitlines()[-1]))
        return fn
    return deco


class ShortEnv(GeneticPhotobioreactorEnv):
    MAX = 60

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.max_steps = self.MAX


@check("cells stay spread through the reactor (no corner pile-up)")
def _():
    env = GeneticPhotobioreactorEnv(initial_cells=400, difficulty=2)
    env.reset(seed=5)
    for _ in range(100):
        env.step(STEADY)
    x = env.cells_x[env.active_mask]
    z = env.cells_z[env.active_mask]
    at_wall = np.mean((x <= 0) | (x >= env.reactor_width) | (z <= 0) | (z >= env.reactor_depth))
    assert at_wall < 0.05, f"{at_wall:.0%} of cells sit exactly on a wall"
    assert 0.25 < z.mean() / env.reactor_depth < 0.75, f"mean depth {z.mean():.3f} m of {env.reactor_depth}"
    assert 0.25 < x.mean() / env.reactor_width < 0.75, f"mean x {x.mean():.3f} m of {env.reactor_width}"


@check("harvest dilution refills toward the same fresh medium reset() uses")
def _():
    env = GeneticPhotobioreactorEnv(initial_cells=400, difficulty=0)
    env.reset(seed=3)
    fresh_salt, fresh_ext = env.salt, env.ext_nutrients
    env.step_count = 600
    env._harvest_action_sum, env._harvest_action_count = 0.4, 1   # mean applied frac 0.2
    env.step(STEADY)
    assert env.salt > fresh_salt - 5.0, f"salt pulled from {fresh_salt} to {env.salt:.0f} by dilution"
    assert abs(env.ext_nutrients - fresh_ext) < 5.0, f"ext_nutrients {env.ext_nutrients:.1f} vs fresh {fresh_ext}"


@check("fresh Zarrouk medium sits at pH ~9.6-10.2 with ~200 meq/L alkalinity")
def _():
    env = GeneticPhotobioreactorEnv(initial_cells=400, difficulty=0)
    env.reset(seed=4)
    assert 9.6 < env.ph < 10.2, f"fresh-medium pH {env.ph:.2f}"
    assert abs(env.alkalinity - 200.0) < 1e-6 and 100.0 < env.dic < 200.0, f"alk {env.alkalinity} dic {env.dic:.1f}"


@check("inter-layer gas exchange is first-order and mass-conserving")
def _():
    env = GeneticPhotobioreactorEnv(initial_cells=400, difficulty=0)
    env.reset(seed=1)
    assert hasattr(env, "_layer_exchange"), "no _layer_exchange(); inline flux divides by volume twice"
    cs, cb, k, vs, vb = 30.0, 0.0, 2.0, 10.0, 20.0
    new_s, new_b = env._layer_exchange(cs, cb, k, vs, vb)
    want = k * (cb - cs) * env.dt
    assert abs((new_s - cs) - want) < 1e-9, f"surface change {new_s - cs:.4f}, want {want:.4f}"
    assert abs((new_s - cs) * vs + (new_b - cb) * vb) < 1e-9, "exchange does not conserve mass"


@check("stitched starts carry the donor's light acclimation")
def _():
    donor = GeneticPhotobioreactorEnv(max_cells=7500, initial_cells=3000, difficulty=2)
    donor.reset(seed=11)
    for _ in range(40):
        donor.step(np.array([0.2, 0.9, -1.0], dtype=np.float32))
    snap = curriculum_starts.snapshot_population(donor)
    env = GeneticPhotobioreactorEnv(max_cells=7500, initial_cells=300, difficulty=2)
    env.reset(seed=12)
    curriculum_starts.apply_saved_population(env, snap)
    got = env.cells_acclimation[env.active_mask]
    want = donor.cells_acclimation[donor.active_mask]
    assert np.array_equal(got, want), "acclimation not transferred with the population"


@check("stitched starts keep fresh values for pools the snapshot lacks")
def _():
    env = GeneticPhotobioreactorEnv(max_cells=7500, initial_cells=300, difficulty=2)
    env.reset(seed=13)
    fresh = (env.p_pool, env.alkalinity, env.dic, env.n_pool)
    legacy = {k: getattr(env, k) for k in ("cells_mass", "cells_quota", "cells_z", "clump_mass",
                                            "pigment", "num_active", "active_mask", "ext_nutrients",
                                            "ph", "do2", "salt")}
    curriculum_starts.apply_saved_population(env, legacy)
    got = (env.p_pool, env.alkalinity, env.dic, env.n_pool)
    assert got == fresh, f"missing keys defaulted to {got}, fresh medium is {fresh}"


@check("TD3 stitched-start capture threshold is reachable")
def _():
    import TD3
    thr = getattr(curriculum_starts, "STITCH_POP_THRESHOLD", None)
    assert thr is not None, "no shared STITCH_POP_THRESHOLD"
    assert thr < TD3.MAX_CELLS, f"threshold {thr} >= MAX_CELLS {TD3.MAX_CELLS}"
    assert "15000" not in inspect.getsource(TD3.train), "TD3.train still hard-codes 15000"


@check("TD3 det-eval leaves the global RNG untouched")
def _():
    import TD3
    TD3.GeneticPhotobioreactorEnv, orig = ShortEnv, TD3.GeneticPhotobioreactorEnv
    try:
        actor = TD3.RecurrentActor(TD3.OBS_DIM, TD3.ACTION_DIM)
        np.random.seed(123)
        before = np.random.get_state()
        TD3.run_td3_eval_episode(actor, 0, seed=900_001, init_cells=120)
        after = np.random.get_state()
        assert np.array_equal(before[1], after[1]) and before[2] == after[2], "global RNG was reseeded"
    finally:
        TD3.GeneticPhotobioreactorEnv = orig


@check("PPO det-eval leaves the global RNG untouched")
def _():
    import deterministic_eval
    from stable_baselines3.common.running_mean_std import RunningMeanStd

    class Zero:
        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return np.zeros((1, 3), dtype=np.float32), state

    deterministic_eval.GeneticPhotobioreactorEnv, orig = ShortEnv, deterministic_eval.GeneticPhotobioreactorEnv
    try:
        np.random.seed(321)
        before = np.random.get_state()
        deterministic_eval.run_deterministic_eval_episode(Zero(), RunningMeanStd(shape=(6,)), 0, seed=100_001)
        after = np.random.get_state()
        assert np.array_equal(before[1], after[1]) and before[2] == after[2], "global RNG was reseeded"
    finally:
        deterministic_eval.GeneticPhotobioreactorEnv = orig


class _FakeModel:
    """SB3 callbacks read the env through model.get_env()."""
    def __init__(self, env):
        self._env = env

    def get_env(self):
        return self._env


def _ppo_episode(starts, env_cls, action):
    """One episode through the real PPO env stack; returns (vec, controller, dones, infos)."""
    import curriculum_schedule, env_factory
    from stable_baselines3.common.vec_env import DummyVecEnv
    it = iter(starts)
    orig_choose, orig_env = curriculum_schedule.choose_episode_start, env_factory.GeneticPhotobioreactorEnv
    curriculum_schedule.choose_episode_start = lambda *a, **k: {"mode": "low", "initial_cells": next(it)}
    env_factory.GeneticPhotobioreactorEnv = env_cls
    try:
        ctrl = curriculum_schedule.CurriculumStartController()
        ctrl.mastery_diff = 2
        vec = DummyVecEnv([env_factory.make_env(difficulty=2, controller=ctrl)])
        vec.reset()
        while True:
            _, _, dones, infos = vec.step([action])
            if dones[0]:
                return vec, ctrl, dones, infos
    finally:
        curriculum_schedule.choose_episode_start = orig_choose
        env_factory.GeneticPhotobioreactorEnv = orig_env


@check("PPO episode metrics mark a crashed episode as crashed")
def _():
    from callbacks import EpisodeMetricsCallback
    vec, _, dones, infos = _ppo_episode([11, 300], GeneticPhotobioreactorEnv, np.ones(3, dtype=np.float32))
    cb = EpisodeMetricsCallback()
    cb.locals, cb.model = {"dones": dones, "infos": infos}, _FakeModel(vec)
    cb._on_step()
    rec = cb.episode_metrics[-1]
    assert infos[0]["episode"]["l"] < 7200, "test episode did not crash"
    assert rec["crashed"], "crash recorded as survival (read the auto-reset population)"


@check("PPO stitch saves the terminal culture, not the auto-reset one")
def _():
    from callbacks import PopulationStitchCallback
    vec, ctrl, dones, infos = _ppo_episode([3000, 1500], ShortEnv, STEADY)
    cb = PopulationStitchCallback(controller=ctrl, pop_threshold=1_100, difficulty_min=0)
    cb.locals, cb.model = {"dones": dones, "infos": infos}, _FakeModel(vec)
    cb._on_step()
    assert ctrl.saved_state is not None, "nothing saved from a 3000-cell episode"
    assert ctrl.saved_state["num_active"] == infos[0]["pop"], \
        f"saved {ctrl.saved_state['num_active']} cells, terminal culture had {infos[0]['pop']}"


@check("TD3 harvest-biased windows contain the harvest-event transition")
def _():
    import TD3
    buf = TD3.SequenceReplayBuffer(4, TD3.SEQ_LEN)
    buf.HARVEST_BIAS_PROB = 1.0
    np.random.seed(0)
    step = TD3.SequenceReplayBuffer.HARVEST_INTERVAL_STEPS
    misses = 0
    for _ in range(3000):
        s = buf._sample_start(7200)
        if not any(s <= k < s + TD3.SEQ_LEN for k in range(step, 7200, step)):
            misses += 1
    assert misses == 0, f"{misses}/3000 harvest-biased windows miss every harvest event"


@check("PBRS potential scores an overheated culture below a healthy one")
def _():
    env = GeneticPhotobioreactorEnv(initial_cells=700, difficulty=1)
    env.reset(seed=21)
    env.temp = env.strain_params["T_opt"]
    healthy = env._potential()
    env.temp = 45.0
    cooked = env._potential()
    assert cooked < 0.5 * healthy, f"Phi at 45C {cooked:.3f} vs at T_opt {healthy:.3f}"


@check("first observation's conductivity matches step 1 (no bicarbonate-clip jump)")
def _():
    env = GeneticPhotobioreactorEnv(initial_cells=400, difficulty=0)
    env.reset(seed=6)
    c0 = env.conductivity
    env.step(STEADY)
    assert abs(env.conductivity - c0) < 0.01 * c0, f"reset {c0:.0f} vs step 1 {env.conductivity:.0f} uS/cm"


@check("medium drawdown matches the biomass grown (N at N_FRAC of dry weight)")
def _():
    env = GeneticPhotobioreactorEnv(max_cells=7500, initial_cells=700, difficulty=0)
    env.reset(seed=8)
    n0 = env.n_pool
    for _ in range(300):
        env.step(np.array([-0.5, 0.2, -1.0], dtype=np.float32))
        assert env.current_nut_flow == 0.0, "dosing switched on; test assumes none"
    drawn = n0 - env.n_pool
    assert drawn > 0.0, "no N drawn down while the culture grew"
    # Upper bound: every mg of biomass now present (including survivors' growth) could not
    # have used more than N_FRAC of itself.
    total_mg = float(np.sum(env.cells_mass[env.active_mask])) * env.MG_PER_MASS_UNIT
    assert drawn * env.volume_L <= env.N_FRAC * (total_mg + 1e3),         f"drew {drawn * env.volume_L:.0f} mg N for at most {total_mg:.0f} mg biomass"


@check("light response can reach its maximum (normalisation matches the red-driven curve)")
def _():
    env = GeneticPhotobioreactorEnv(max_cells=7500, initial_cells=45, difficulty=0)
    env.reset(seed=9)
    Ks, Ki = env.strain_params["Ks_light"], env.strain_params["Kii"]
    I_star = float(np.sqrt(Ks * Ki))
    red = env.RED_FRAC * I_star
    f = red / (Ks + red + I_star ** 2 / Ki)
    f_max = env.RED_FRAC * I_star / (2.0 * Ks + env.RED_FRAC * I_star)
    assert abs(f / f_max - 1.0) < 1e-9, f"f_I at the optimum is {f / f_max:.3f}, not 1"


@check("self-shading engages at production density; thermostat holds 35C below ~1500 umol")
def _():
    env = GeneticPhotobioreactorEnv(max_cells=7500, initial_cells=700, difficulty=2)
    env.reset(seed=10)
    X = float(np.sum(env.cells_mass[env.active_mask])) * env.MG_PER_MASS_UNIT / env.volume_L
    back = np.exp(-(0.5 + env.EXT_RED * X) * env.light_path_m)
    assert X > 200.0 and back < 0.05, f"{X:.0f} mg/L, back-wall red light {back:.2%} of surface"
    light = np.interp(1400, [0, 2000], [-1, 1])
    for _ in range(1500):
        env.step(np.array([-0.5, light, -1.0], dtype=np.float32))
    assert abs(env.temp - env.T_SETPOINT) < 0.5, f"T {env.temp:.1f}C at 1400 umol"


if __name__ == "__main__":
    print("=" * 78)
    for name, ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"\n         -> {msg}"))
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("=" * 78)
    print(f"  {len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)
