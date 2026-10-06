"""PC-Gym CSTR task (experiments/pcgym_control/tasks.py): scoring maths, runaway flag, scenario
generator, controller-failure handling, and a golden check against the committed reference PID."""
import json
import os

import numpy as np
import pytest

from _ppo_helpers import PG, PG_FINAL, load_module

tasks = load_module("pg_tasks_under_test", os.path.join(PG, "tasks.py"))
CSTR = tasks.TASKS["cstr"]


class FakeEnv:
    """Replays a prescribed (Ca, T) trajectory instead of integrating the CSTR ODE."""

    def __init__(self, ca, temp, sp):
        self.ca, self.temp, self.sp, self.k, self.actions = ca, temp, sp, 0, []

    def step(self, action):
        self.actions.append(float(action[0]))
        state = np.array([self.ca[self.k], self.temp[self.k], self.sp[self.k]])
        self.k += 1
        return state, 0.0, False, False, {}


def scripted_episode(seed, ca_error, temp):
    """A real episode object (so scenario / setpoints are the genuine ones) driven by a fake plant."""
    ep = CSTR.start(seed)
    sp = ep.sc["sp"]
    ep.env = FakeEnv(sp + ca_error, temp, sp)
    return ep


def run_all(ep, u=298.0):
    while not ep.done:
        ep.step(u)
    return ep.summary()


# ------------------------------------------------------------------ cost --------------------------
def test_cost_is_mean_squared_error_in_units_of_0p01():
    err = np.zeros(CSTR.N)
    err[:40] = 0.01          # exactly 1 unit  -> contributes 1 each
    err[40:50] = -0.02       # 2 units         -> 4 each
    err[50:60] = 0.005       # 0.5 units       -> 0.25 each
    ep = scripted_episode(20_000_000, err, np.full(CSTR.N, 325.0))
    s = run_all(ep)
    expected = np.mean((err / 0.01) ** 2)
    assert expected == pytest.approx((40 * 1 + 10 * 4 + 10 * 0.25) / 120)
    assert s["cost"] == pytest.approx(expected, rel=1e-6)
    assert s["seed"] == 20_000_000


def test_cost_scores_ca_against_the_setpoint_of_the_step_just_taken():
    """With a stepped setpoint, a plant sitting on the OLD level just after the change is penalised."""
    ep = CSTR.start(20_000_001)
    sp = ep.sc["sp"]
    assert len(set(sp)) > 1                                            # the scenario really steps
    ca = np.full(CSTR.N, sp[0])                                        # plant never leaves level 1
    ep.env = FakeEnv(ca, np.full(CSTR.N, 325.0), sp)
    s = run_all(ep)
    expected = np.mean(((ca - sp) / 0.01) ** 2)
    assert expected > 0
    assert s["cost"] == pytest.approx(expected, rel=1e-6)


def test_perfect_tracking_costs_zero():
    ep = scripted_episode(7, np.zeros(CSTR.N), np.full(CSTR.N, 320.0))
    assert run_all(ep)["cost"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ runaway -----------------------
@pytest.mark.parametrize("peak,runaway", [(334.99, False), (335.0, False), (335.01, True), (360.0, True)])
def test_runaway_flag_is_strictly_above_335_K(peak, runaway):
    assert CSTR.T_RUNAWAY == 335.0
    temp = np.full(CSTR.N, 325.0)
    temp[77] = peak
    ep = scripted_episode(3, np.zeros(CSTR.N), temp)
    x0_T = ep.t_max
    s = run_all(ep)
    assert s["runaway"] is runaway
    assert s["t_max"] == pytest.approx(max(peak, x0_T))


def test_t_max_includes_the_initial_temperature():
    ep = scripted_episode(3, np.zeros(CSTR.N), np.full(CSTR.N, 300.0))
    x0_T = ep.sc["x0"][1]
    assert run_all(ep)["t_max"] == pytest.approx(x0_T)


# ------------------------------------------------------------------ actuator / bookkeeping --------
def test_jacket_temperature_is_clipped_to_the_actuator_range():
    ep = scripted_episode(5, np.zeros(CSTR.N), np.full(CSTR.N, 325.0))
    assert ep.step(310.0) == 302.0
    assert ep.step(280.0) == 295.0
    assert ep.step(299.5) == 299.5
    assert ep.env.actions == [302.0, 295.0, 299.5]


def test_episode_ends_after_n_steps():
    ep = scripted_episode(5, np.zeros(CSTR.N), np.full(CSTR.N, 325.0))
    for _ in range(CSTR.N - 1):
        ep.step(298.0)
        assert not ep.done
    ep.step(298.0)
    assert ep.done and ep.k == CSTR.N


# ------------------------------------------------------------------ scenario ----------------------
def test_scenario_is_deterministic_in_the_seed_and_within_documented_ranges():
    a, b = CSTR.scenario(20_000_000), CSTR.scenario(20_000_000)
    for k in ("sp", "Ti", "Caf", "x0"):
        np.testing.assert_array_equal(a[k], b[k])
    assert a["noise_seed"] == b["noise_seed"]
    for seed in range(20_000_000, 20_000_030):
        sc = CSTR.scenario(seed)
        assert len(sc["sp"]) == len(sc["Ti"]) == len(sc["Caf"]) == CSTR.N
        assert CSTR.SP_RANGE[0] <= sc["sp"].min() and sc["sp"].max() <= CSTR.SP_RANGE[1]
        assert CSTR.TI_RANGE[0] <= sc["Ti"].min() and sc["Ti"].max() <= CSTR.TI_RANGE[1]
        assert CSTR.CAF_RANGE[0] <= sc["Caf"].min() and sc["Caf"].max() <= CSTR.CAF_RANGE[1]
        assert 0.85 <= sc["x0"][0] <= 0.91 and 320.0 <= sc["x0"][1] <= 326.0
        # setpoint holds each level for >= 20 steps
        runs = np.diff(np.flatnonzero(np.concatenate([[True], np.diff(sc["sp"]) != 0, [True]])))
        assert runs.min() >= 20
    assert not np.array_equal(CSTR.scenario(1)["sp"], CSTR.scenario(2)["sp"])


def test_scenarios_do_not_depend_on_the_global_numpy_state():
    np.random.seed(0)
    a = CSTR.scenario(42)["sp"]
    np.random.seed(999)
    b = CSTR.scenario(42)["sp"]
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ run(): controller failures -----
class Raises:
    def act(self, obs):
        raise RuntimeError("boom")


class NaNController:
    def act(self, obs):
        return float("nan")


@pytest.mark.parametrize("ctrl,msg", [(Raises(), "RuntimeError: boom"), (NaNController(), "non-finite")])
def test_failing_controller_is_scored_with_jacket_held_mid_range(ctrl, msg):
    summary, trace = CSTR.run(ctrl, 20_000_000)
    assert msg in summary["error"]
    assert len(trace) == CSTR.N and all(row["Tc"] == 298.5 for row in trace)
    assert np.isfinite(summary["cost"])


def test_observation_exposes_only_noisy_measurements_and_setpoint():
    obs = CSTR.start(20_000_000).obs()
    assert set(obs) == {"t_min", "Ca", "T", "Ca_sp"}
    summary, trace = CSTR.run(type("C", (), {"act": lambda self, o: 298.0})(), 20_000_000, privileged=False)
    assert "Ca_true" not in trace[0]
    _, trace_p = CSTR.run(type("C", (), {"act": lambda self, o: 298.0})(), 20_000_000, privileged=True)
    assert {"Ca_true", "T_true", "Ti", "Caf"} <= set(trace_p[0])


# ------------------------------------------------------------------ golden vs committed reference --
def test_reference_pid_reproduces_committed_final_episodes():
    """ref-pid__only.json holds controllers/pid.py (default gains) on the final split."""
    committed = json.load(open(os.path.join(PG_FINAL, "ref-pid__only.json")))["episodes"]
    pid = load_module("pg_pid_under_test", os.path.join(PG, "controllers", "pid.py"))
    for e in committed[:3]:
        s, _ = CSTR.run(pid.Controller(), e["seed"])
        assert s["cost"] == pytest.approx(e["cost"], rel=1e-9)
        assert s["runaway"] == e["runaway"]
        assert s["t_max"] == pytest.approx(e["t_max"], rel=1e-9)
