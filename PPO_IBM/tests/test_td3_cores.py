"""TD3 recurrent cores: checkpoint detection / round trip, hidden-state reset cadence, LRU state
bounds, and the program-harness adapter (controllers/td3_actor.py). CPU only, no training."""
import os

import numpy as np
import pytest
import torch

import TD3
import TD3_cores
import TD3_lru
import actor_io
from lru_core import LRUCore
from rtu_core import GRUCore, RTUCore

from _ppo_helpers import PC, TD3_DIR, load_module

CPU = torch.device("cpu")
OBS_DIM, ACTION_DIM = TD3.OBS_DIM, TD3.ACTION_DIM
GRU_ACTOR, GRU_CRITIC = TD3_cores.make_classes("gru")
RTU_ACTOR, RTU_CRITIC = TD3_cores.make_classes("rtu")

ACTORS = {"lstm": TD3.RecurrentActor, "lru": TD3_lru.LRUActor, "gru": GRU_ACTOR, "rtu": RTU_ACTOR}
CRITICS = {"lstm": TD3.RecurrentCritic, "lru": TD3_lru.LRUCritic, "gru": GRU_CRITIC, "rtu": RTU_CRITIC}


def fixed_obs(batch=1, steps=8):
    g = torch.Generator().manual_seed(0)
    return torch.randn(batch, steps, OBS_DIM, generator=g)


# ------------------------------------------------------------------ detect_core / load_actor ------
@pytest.mark.parametrize("core", sorted(ACTORS))
def test_detect_core_on_actor_and_critic_state_dicts(core):
    torch.manual_seed(1)
    assert actor_io.detect_core(ACTORS[core](OBS_DIM, ACTION_DIM).state_dict()) == core
    assert actor_io.detect_core(CRITICS[core](OBS_DIM, ACTION_DIM).state_dict()) == core


def test_detect_core_rejects_unknown_checkpoints():
    with pytest.raises(ValueError, match="Cannot identify the recurrent core"):
        actor_io.detect_core({"foo.weight": torch.zeros(1), "bar.bias": torch.zeros(1)})


@pytest.mark.parametrize("core", sorted(ACTORS))
def test_load_actor_round_trip_gives_identical_outputs(core, tmp_path):
    torch.manual_seed(2)
    actor = ACTORS[core](OBS_DIM, ACTION_DIM).to(CPU).eval()
    path = tmp_path / f"actor_{core}.pth"
    torch.save(actor.state_dict(), path)

    loaded, detected = actor_io.load_actor(str(path), device=CPU)
    assert detected == core
    assert type(loaded).__name__ == type(actor).__name__
    assert not loaded.training

    x = fixed_obs()
    with torch.no_grad():
        a_ref, h_ref = actor(x, actor.initial_hidden(batch=1))
        a_new, h_new = loaded(x, loaded.initial_hidden(batch=1))
    assert torch.equal(a_ref, a_new)
    assert a_ref.shape == (1, 8, ACTION_DIM) and a_ref.abs().max() <= 1.0       # tanh-squashed
    for r, n in zip(h_ref if isinstance(h_ref, tuple) else (h_ref,), h_new if isinstance(h_new, tuple) else (h_new,)):
        assert torch.equal(r, n)


@pytest.mark.parametrize("core", sorted(ACTORS))
def test_load_critic_round_trip(core, tmp_path):
    torch.manual_seed(3)
    critic = CRITICS[core](OBS_DIM, ACTION_DIM).to(CPU).eval()
    path = tmp_path / f"critic_{core}.pth"
    torch.save(critic.state_dict(), path)
    loaded, detected = actor_io.load_critic(str(path), device=CPU)
    assert detected == core
    obs, act = fixed_obs(), torch.zeros(1, 8, ACTION_DIM)
    with torch.no_grad():
        q_ref = critic(obs, act)[:2]
        q_new = loaded(obs, act)[:2]
    assert torch.equal(q_ref[0], q_new[0]) and torch.equal(q_ref[1], q_new[1])


@pytest.mark.parametrize("core", sorted(ACTORS))
def test_stepwise_rollout_matches_sequence_forward(core):
    """The rollout path (T = 1, carrying hidden) must equal the training path (whole window)."""
    torch.manual_seed(4)
    actor = ACTORS[core](OBS_DIM, ACTION_DIM).eval()
    x = fixed_obs(steps=12)
    with torch.no_grad():
        full, _ = actor(x, actor.initial_hidden(batch=1))
        h, outs = actor.initial_hidden(batch=1), []
        for t in range(12):
            a, h = actor(x[:, t:t + 1], h)
            outs.append(a)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------ hidden reset interval ---------
def _hidden_reset_interval(monkeypatch, env_value):
    """Execute a private copy of TD3.py under the given environment (the value is read at import)."""
    if env_value is None:
        monkeypatch.delenv("TD3_HIDDEN_RESET_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("TD3_HIDDEN_RESET_INTERVAL", env_value)
    fresh = load_module("TD3_private_copy", os.path.join(TD3_DIR, "TD3.py"))
    return fresh.HIDDEN_RESET_INTERVAL, fresh.SEQ_LEN


def test_hidden_reset_interval_is_read_from_the_environment_at_import(monkeypatch):
    assert _hidden_reset_interval(monkeypatch, None) == (60, 60)       # default: SEQ_LEN, unchanged behaviour
    assert _hidden_reset_interval(monkeypatch, "600") == (600, 60)     # override decoupled from SEQ_LEN
    assert _hidden_reset_interval(monkeypatch, "60") == (60, 60)


class CountingActor:
    """Stands in for an actor: hidden state is a counter of steps since the last reset."""

    def __init__(self):
        self.resets, self.seen = 0, []

    def initial_hidden(self, batch):
        self.resets += 1
        return torch.zeros(1)

    def eval(self): return self

    def train(self): return self

    def __call__(self, obs, hidden):
        self.seen.append(int(hidden.item()))
        return torch.zeros(1, 1, ACTION_DIM), hidden + 1


class FakeEnv:
    max_steps = 23

    def __init__(self, **kw):
        self.t = 0

    def reset(self, seed=None):
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, self.t >= self.max_steps, {}


def test_eval_rollout_resets_hidden_state_every_interval(monkeypatch):
    monkeypatch.setattr(TD3, "GeneticPhotobioreactorEnv", FakeEnv)
    monkeypatch.setattr(TD3, "HIDDEN_RESET_INTERVAL", 5)
    actor = CountingActor()
    res = TD3._run_td3_eval_episode(actor, difficulty=2, seed=1, init_cells=100)
    assert actor.seen == [0, 1, 2, 3, 4] * 4 + [0, 1, 2]       # fresh state at steps 0, 5, 10, 15, 20
    assert actor.resets == 1 + 4                                # initial + one reset per elapsed interval
    assert res["crashed"] is False


def test_eval_rollout_free_running_when_interval_exceeds_episode(monkeypatch):
    monkeypatch.setattr(TD3, "GeneticPhotobioreactorEnv", FakeEnv)
    monkeypatch.setattr(TD3, "HIDDEN_RESET_INTERVAL", 600)
    actor = CountingActor()
    TD3._run_td3_eval_episode(actor, difficulty=0, seed=1, init_cells=100)
    assert actor.seen == list(range(23)) and actor.resets == 1


# ------------------------------------------------------------------ LRU / RTU state ---------------
def test_lru_state_bounded_over_long_random_input():
    torch.manual_seed(5)
    core = LRUCore(128).eval()
    _, lam, gamma = core._decay()
    g = torch.Generator().manual_seed(6)
    h = core.initial_hidden(1)
    peak, max_u = torch.zeros(128), torch.zeros(128)
    with torch.no_grad():
        for _ in range(80):                                           # 80 x 250 = 20,000 steps
            x = 5.0 * torch.randn(1, 250, 128, generator=g)           # deliberately large inputs
            max_u = torch.maximum(max_u, core.in_proj(x)[0].abs().max(dim=0).values)
            out, h = core(x, h)
            assert torch.isfinite(h).all() and torch.isfinite(out).all()
            peak = torch.maximum(peak, h[0].abs())
    assert torch.all(lam > 0) and torch.all(lam < 1)
    assert torch.all(peak <= gamma * max_u / (1 - lam) + 1e-4)       # |h| <= gamma * max|u| / (1 - lam)
    assert peak.max() < 1e3                                           # nowhere near saturating / blowing up


def test_lru_converges_to_its_fixed_point_on_constant_input():
    torch.manual_seed(7)
    core = LRUCore(32, r_min=0.6, r_max=0.9).eval()                  # fast decay so 400 steps is plenty
    _, lam, gamma = core._decay()
    x = torch.randn(1, 32)
    h = core.initial_hidden(1)
    with torch.no_grad():
        for _ in range(400):
            _, h = core.step(x, h)
        fixed = gamma * core.in_proj(x)[0] / (1 - lam)
    torch.testing.assert_close(h[0], fixed, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("make_core", [lambda: LRUCore(32), lambda: RTUCore(32), lambda: GRUCore(32)],
                         ids=["lru", "rtu", "gru"])
def test_core_chunked_forward_equals_whole_sequence(make_core):
    """Carrying the hidden state across calls is lossless: the property the reset interval relies on."""
    torch.manual_seed(8)
    core = make_core().eval()
    x = torch.randn(2, 20, 32)
    with torch.no_grad():
        full, h_full = core(x, core.initial_hidden(2))
        o1, h1 = core(x[:, :9], core.initial_hidden(2))
        o2, h2 = core(x[:, 9:], h1)
    torch.testing.assert_close(torch.cat([o1, o2], dim=1), full, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(h2, h_full, atol=1e-5, rtol=1e-4)


def test_rtu_state_bounded_over_long_input():
    torch.manual_seed(9)
    core = RTUCore(32).eval()
    h = core.initial_hidden(1)
    g = torch.Generator().manual_seed(10)
    with torch.no_grad():
        for _ in range(5_000):
            _, h = core.step(3.0 * torch.randn(1, 32, generator=g), h)
    assert torch.isfinite(h).all() and h.abs().max() < 1e3


# ------------------------------------------------------------------ program-harness adapter -------
OBS = {"turbidity_ntu": 150.0, "ph": 9.9, "pump_L": 0.0, "conductivity": 29500.0, "temp_c": 36.0, "lux": 580.0}


class Spy:
    def __init__(self, actor):
        self.actor, self.resets = actor, 0

    def initial_hidden(self, batch):
        self.resets += 1
        return self.actor.initial_hidden(batch)

    def __call__(self, x, hidden):
        return self.actor(x, hidden)


@pytest.fixture
def adapter(tmp_path):
    torch.manual_seed(11)
    path = tmp_path / "actor.pth"
    torch.save(TD3_lru.LRUActor(OBS_DIM, ACTION_DIM).state_dict(), path)
    mod = load_module("td3_actor_adapter_under_test", os.path.join(PC, "controllers", "td3_actor.py"))
    return mod, str(path)


def test_adapter_reset_parameter_controls_hidden_state_resets(adapter):
    mod, path = adapter
    ctrl = mod.Controller({"actor_path": path, "reset": 3})
    assert ctrl.reset == 3
    ctrl.actor = Spy(ctrl.actor)
    outs = [ctrl.act(dict(OBS)) for _ in range(7)]
    assert ctrl.actor.resets == 2                              # before the 4th and the 7th call
    np.testing.assert_allclose(outs[3], outs[0], atol=1e-6)     # state zeroed again -> same output as call 1
    np.testing.assert_allclose(outs[6], outs[0], atol=1e-6)
    assert not np.allclose(outs[1], outs[0], atol=1e-6)         # and the state does matter in between


def test_adapter_default_reset_is_600_and_free_runs_before_it(adapter):
    mod, path = adapter
    ctrl = mod.Controller({"actor_path": path})
    assert ctrl.reset == 600
    ctrl.actor = Spy(ctrl.actor)
    for _ in range(50):
        ctrl.act(dict(OBS))
    assert ctrl.actor.resets == 0


def test_adapter_actions_are_in_physical_ranges_and_cached_actor_is_shared(adapter):
    mod, path = adapter
    c1, c2 = mod.Controller({"actor_path": path}), mod.Controller({"actor_path": path})
    assert c1.actor is c2.actor                                  # checkpoint loaded once per path
    stir, light, frac = c1.act(dict(OBS))
    assert 50 <= stir <= 200 and 0 <= light <= 2000 and 0 <= frac <= 0.5
    assert all(isinstance(v, float) for v in (stir, light, frac))
