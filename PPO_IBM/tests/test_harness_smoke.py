"""Smoke tests for the program_control harness: it loads a controller program, runs episodes, and
scores them. Episodes are shortened by patching the env's max_steps (run in-process, workers=1), so the
default run stays fast; one full-length golden episode against the committed reference is marked slow."""
import json
import os
import textwrap

import numpy as np
import pytest

import genetic_env
from _ppo_helpers import PC, PC_FINAL, load_module

h = load_module("pc_harness_smoke_under_test", os.path.join(PC, "harness.py"))


def shorten(monkeypatch, n_steps):
    """Cap every env built from now on at n_steps (the env calls super(Cls, self), so patch __init__
    in place rather than subclassing)."""
    cls = genetic_env.GeneticPhotobioreactorEnv
    orig = cls.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.max_steps = n_steps

    monkeypatch.setattr(cls, "__init__", init)


@pytest.fixture
def short_episodes(monkeypatch):
    """Episodes of 650 steps: long enough to include the first harvest event at step 600."""
    shorten(monkeypatch, 650)
    return 650


def write_controller(tmp_path, body, name="ctrl.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


CONSTANT = """
class Controller:
    def __init__(self, params=None):
        self.p = dict(stir=100.0, light=900.0, frac=0.2)
        self.p.update(params or {})
    def act(self, obs):
        return self.p["stir"], self.p["light"], self.p["frac"]     # the env harvests the 600-step MEAN of frac
"""


# ------------------------------------------------------------------ action mapping ----------------
def test_to_env_action_maps_physical_ranges_to_unit_box():
    f = h.to_env_action
    np.testing.assert_allclose(f(50, 0, 0, 0.5), [-1, -1, -1])
    np.testing.assert_allclose(f(200, 2000, 0.5, 0.5), [1, 1, 1])
    np.testing.assert_allclose(f(125, 1000, 0.25, 0.5), [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(f(-10, 5000, 9.0, 0.5), [-1, 1, 1])           # clipped, not extrapolated
    assert f(100, 100, 0.1, 0.5).dtype == np.float32


# ------------------------------------------------------------------ run_episode / evaluate --------
def test_trivial_controller_is_scored_on_two_short_episodes(tmp_path, short_episodes):
    path = write_controller(tmp_path, CONSTANT)
    results = [h.run_episode((path, None, 2, ic, seed, False, 0)) for ic, seed in [(200, 1234), (350, 1235)]]
    for r, (ic, seed) in zip(results, [(200, 1234), (350, 1235)]):
        assert (r["init_cells"], r["seed"]) == (ic, seed)
        assert r["steps"] == 650 and r["crashed"] is False and r["error"] is None
        assert r["harvested_mg"] > 0                       # the 20% harvest at step 600 removed biomass
        assert np.isfinite(r["time_avg_od"])
    s = h.summarise(results)
    assert s["n"] == 2 and s["errors"] == 0 and s["crash_rate"] == 0.0
    assert s["fitness"] == pytest.approx(0.5 * s["median_mg"] + 0.5 * s["p25_mg"])


def test_run_episode_is_reproducible_for_a_given_seed(tmp_path, short_episodes):
    path = write_controller(tmp_path, CONSTANT)
    job = (path, None, 1, 250, 777, False, 0)
    a, b = h.run_episode(job), h.run_episode(job)
    assert a == b


def test_controller_params_reach_the_program(tmp_path, short_episodes):
    path = write_controller(tmp_path, CONSTANT)
    lo = h.run_episode((path, {"frac": 0.05}, 0, 300, 5, False, 0))
    hi = h.run_episode((path, {"frac": 0.45}, 0, 300, 5, False, 0))
    assert hi["harvested_mg"] > lo["harvested_mg"]


def test_evaluate_runs_the_whole_search_split_with_one_worker(tmp_path, monkeypatch):
    shorten(monkeypatch, 40)
    path = write_controller(tmp_path, CONSTANT)
    summary, results = h.evaluate(path, split="search", difficulty=2, workers=1)
    assert summary["n"] == 12 == len(results)
    assert [(r["init_cells"], r["seed"]) for r in results] == h.split_jobs("search", 2)
    assert all(r["steps"] == 40 and r["error"] is None for r in results)


@pytest.mark.parametrize("body,expect", [
    ("class Controller:\n    def act(self, obs):\n        raise ValueError('nope')\n", "ValueError: nope"),
    ("class Controller:\n    def act(self, obs):\n        return float('nan'), 100.0, 0.0\n", "non-finite"),
    ("raise ImportError('missing dependency')\n", "at load"),
])
def test_broken_programs_forfeit_the_episode_and_are_flagged(tmp_path, short_episodes, body, expect):
    path = write_controller(tmp_path, body)
    r = h.run_episode((path, None, 2, 200, 9, False, 0))
    assert expect in r["error"]
    assert r["crashed"] is True and r["steps"] < 650
    s = h.summarise([r])
    assert s["errors"] == 1 and s["crash_rate"] == 1.0
    assert s["fitness"] <= -10_000.0 + 1e-9               # a forfeit is charged like a crash


def test_privileged_flag_exposes_true_od_only_when_asked(tmp_path, short_episodes):
    asserts_hidden = write_controller(tmp_path, """
        class Controller:
            def act(self, obs):
                assert "true_od" not in obs
                return 100.0, 900.0, 0.0
    """, name="hidden.py")
    asserts_visible = write_controller(tmp_path, """
        class Controller:
            def act(self, obs):
                assert obs["true_od"] > 0
                return 100.0, 900.0, 0.0
    """, name="visible.py")
    assert h.run_episode((asserts_hidden, None, 2, 200, 9, False, 0))["error"] is None
    assert h.run_episode((asserts_visible, None, 2, 200, 9, True, 0))["error"] is None
    assert h.run_episode((asserts_visible, None, 2, 200, 9, False, 0))["error"] is not None


def test_summarise_excludes_adversarial_episodes_from_yield_but_not_from_crash_rate():
    base = dict(steps=7200, error=None, time_avg_od=0.5, seed=0)
    res = [dict(base, init_cells=50, harvested_mg=1.0, crashed=True),
           dict(base, init_cells=200, harvested_mg=10_000.0, crashed=False),
           dict(base, init_cells=300, harvested_mg=20_000.0, crashed=False)]
    s = h.summarise(res)
    assert s["median_mg"] == pytest.approx(15_000.0)
    assert s["crash_rate"] == pytest.approx(1 / 3)


# ------------------------------------------------------------------ slow golden -------------------
@pytest.mark.slow
def test_reference_sensor_expert_reproduces_a_committed_final_episode():
    """One full-length (7200-step) final-split episode of the hand-written sensor expert."""
    committed = json.load(open(os.path.join(PC_FINAL, "ref-expert__only.json")))["episodes"][1]
    path = os.path.join(PC, "controllers", "sensor_expert.py")
    r = h.run_episode((path, None, 2, committed["init_cells"], committed["seed"], False, 0))
    assert r["steps"] == committed["steps"] == 7200
    assert r["harvested_mg"] == pytest.approx(committed["harvested_mg"], rel=1e-6)
    assert r["time_avg_od"] == pytest.approx(committed["time_avg_od"], rel=1e-6)
