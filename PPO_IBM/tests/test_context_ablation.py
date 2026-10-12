"""Training-context ablation of the TD3 recurrent actors: TD3_SEQ_LEN, TD3_BURN_IN (+ stored actor
state), TD3_CORE_LN, and the 'val' split. The defaults must reproduce rl-protocol-v1 exactly."""
import ast
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import TD3
import TD3_cores
import TD3_lru
import actor_io
from ln_lstm import LNLSTM

import _td3_ref
from _ppo_helpers import PC, TD3_DIR, load_module

HERE = os.path.dirname(os.path.abspath(__file__))
OBS, ACT = TD3.OBS_DIM, TD3.ACTION_DIM
GRU_A, GRU_C = TD3_cores.make_classes("gru")
RTU_A, RTU_C = TD3_cores.make_classes("rtu")
CORES = {"lstm": (TD3.RecurrentActor, TD3.RecurrentCritic), "lru": (TD3_lru.LRUActor, TD3_lru.LRUCritic),
         "gru": (GRU_A, GRU_C), "rtu": (RTU_A, RTU_C)}


# ------------------------------------------------------------------ 1. default-path identity -----
@pytest.mark.parametrize("core", ["lstm", "lru"])
def test_defaults_reproduce_v1_update_sequence(core):
    """The fixture was produced by td3/TD3.py + TD3_lru.py as of rl-protocol-v1 (before any flag
    existed): 8 seeded td3_update calls on synthetic episodes. With every flag at its default the
    parameters must match it (bit-identical on the reference machine; the tolerance only covers
    BLAS differences elsewhere)."""
    assert (TD3.SEQ_LEN, TD3.BURN_IN, TD3.CORE_LN) == (60, 0, False)
    ref = np.load(os.path.join(HERE, "fixtures", "td3_default_ref.npz"))
    out = _td3_ref.run(core, TD3, TD3_lru)
    assert out
    for k, v in out.items():
        np.testing.assert_allclose(v, ref[f"{core}/{k}"], rtol=1e-4, atol=1e-5, err_msg=k)


def test_flag_parsing_in_a_fresh_process():
    code = ("import TD3; print(TD3.SEQ_LEN, TD3.BURN_IN, TD3.STATE_STRIDE, TD3.CORE_LN, TD3.HIDDEN_RESET_INTERVAL)")
    env = dict(os.environ, TD3_SEQ_LEN="150", TD3_BURN_IN="40", TD3_STATE_STRIDE="10", TD3_CORE_LN="1",
               PYTHONPATH=os.pathsep.join([TD3_DIR, os.path.join(os.path.dirname(HERE), "environments"),
                                           os.path.join(os.path.dirname(HERE), "training")]))
    env.pop("TD3_HIDDEN_RESET_INTERVAL", None)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=HERE)
    # the reset cadence does not follow TD3_SEQ_LEN
    assert r.stdout.split() == ["150", "40", "10", "True", "60"], r.stderr[-400:]


# ------------------------------------------------------------------ 2. burn-in sampling ----------
def episode_with_states(n=1300, stride=20, d=8, reset=600):
    ep = _td3_ref.synthetic_episode(np.random.RandomState(0), n)
    # distinct non-zero state per stored index, zero at reset boundaries (as a rollout would hold)
    st = np.stack([np.full(d, float(i + 1), dtype=np.float16) for i in range(0, n, stride)])
    for i in range(0, n, stride):
        if i % reset == 0:
            st[i // stride] = 0
    ep["hstate"], ep["hstride"] = st, stride
    return ep


def test_burn_info_indices(monkeypatch):
    monkeypatch.setattr(TD3, "BURN_IN", 40)
    monkeypatch.setattr(TD3, "HIDDEN_RESET_INTERVAL", 600)
    ep = episode_with_states()
    buf = TD3.SequenceReplayBuffer(4, 60)
    for start in range(0, 1300 - 60 + 1):
        info = buf.burn_info(ep, start)
        b, r0 = info["b"], (start // 600) * 600
        assert b % 20 == 0 and b <= start and b >= r0
        # largest stored index <= start - BURN_IN, except where the last hidden reset cuts it short
        assert b <= start - 40 or b == r0
        assert b + 20 > start - 40 or b == 0 or b == r0
        assert info["obs"].shape[0] == start - b + 1
        np.testing.assert_array_equal(info["obs"], ep["obs"][b:start + 1])
        np.testing.assert_array_equal(info["h0"], ep["hstate"][b // 20].astype(np.float32))


def test_burn_info_zero_state_fallbacks(monkeypatch):
    monkeypatch.setattr(TD3, "BURN_IN", 40)
    monkeypatch.setattr(TD3, "HIDDEN_RESET_INTERVAL", 600)
    buf = TD3.SequenceReplayBuffer(4, 60)
    ep = episode_with_states()
    for start in (0, 7, 39):                       # start < BURN_IN: burn-in is whatever prefix exists
        info = buf.burn_info(ep, start)
        assert info["b"] == 0 and not np.any(info["h0"])
    demo = _td3_ref.synthetic_episode(np.random.RandomState(1), 700)      # no stored states
    for start in (0, 30, 333, 640):
        info = buf.burn_info(demo, start)
        assert info["h0"] is None and info["b"] % 20 == 0 and info["b"] <= max(0, start - 40)


def test_sample_mixed_batch_carries_burn_info_only_when_enabled(monkeypatch):
    demo = TD3.SequenceReplayBuffer(4, 60)
    online = TD3.SequenceReplayBuffer(4, 60)
    rng = np.random.RandomState(3)
    for _ in range(3):
        demo.add(_td3_ref.synthetic_episode(rng, 300))
    online.add(episode_with_states(300))
    assert "burn" not in TD3.sample_mixed_batch(demo, online, 8, 0.25)
    monkeypatch.setattr(TD3, "BURN_IN", 40)
    batch = TD3.sample_mixed_batch(demo, online, 8, 0.25)
    assert len(batch["burn"]) == batch["obs"].shape[0] == 8
    assert sum(it["h0"] is None for it in batch["burn"]) >= 2          # the demo rows


# ------------------------------------------------------------------ 3. stored state == rollout state
def rollout(actor, obs, reset, stride):
    rec = TD3.StateRecorder(stride)
    hidden, since = actor.initial_hidden(batch=1), 0
    with torch.no_grad():
        for t in range(len(obs)):
            if since >= reset:
                hidden, since = actor.initial_hidden(batch=1), 0
            rec.before_step(t, hidden)
            _, hidden = actor(torch.tensor(obs[t], dtype=torch.float32).view(1, 1, -1), hidden)
            since += 1
    return rec


@pytest.mark.parametrize("core", ["lstm", "lru", "gru", "rtu"])
def test_stored_state_equals_actual_rollout_state(core):
    torch.manual_seed(5)
    actor = CORES[core][0](OBS, ACT).eval()
    n, reset, stride = 260, 100, 20
    obs = np.random.RandomState(6).randn(n, OBS).astype(np.float32) * 3
    rec = rollout(actor, obs, reset, stride)
    assert len(rec.states) == len(range(0, n, stride)) and rec.states[0].dtype == np.float16
    with torch.no_grad():
        for k, st in enumerate(rec.states):
            idx = k * stride
            r0 = (idx // reset) * reset
            if idx == r0:
                expect = torch.zeros(1, st.shape[0])
            else:   # independent route: one batched forward from the last reset
                _, h = actor(torch.tensor(obs[r0:idx]).unsqueeze(0), actor.initial_hidden(1))
                expect = TD3.hidden_to_vec(h)
            np.testing.assert_allclose(st.astype(np.float32), expect.reshape(-1).numpy(), rtol=3e-3, atol=2e-3,
                                       err_msg=f"{core} idx {idx}")


@pytest.mark.parametrize("core", ["lstm", "lru"])
def test_burn_in_context_matches_a_full_run_from_the_reset(core, monkeypatch):
    """The whole chain (rollout storage -> sampling -> grouped burn-in) must hand the window the
    state the actor had, for the online actor and, with the same stored state, its target."""
    monkeypatch.setattr(TD3, "BURN_IN", 40)
    monkeypatch.setattr(TD3, "HIDDEN_RESET_INTERVAL", 100)
    torch.manual_seed(7)
    A, C = CORES[core]
    actor, actor_t, critic, critic_t = A(OBS, ACT).eval(), A(OBS, ACT).eval(), C(OBS, ACT), C(OBS, ACT)
    n = 400
    ep = _td3_ref.synthetic_episode(np.random.RandomState(8), n)
    rec = rollout(actor, ep["obs"], 100, 20)
    rec.attach(ep)
    buf = TD3.SequenceReplayBuffer(4, 60)
    buf.add(ep)
    np.random.seed(0)
    batch = TD3.sample_mixed_batch(TD3.SequenceReplayBuffer(1, 60), buf, 12, 0.0)
    ctx = TD3.burn_in_context(batch, actor, actor_t, critic, critic_t)
    vec = ctx["actor_vec"]
    for i, it in enumerate(batch["burn"]):
        s = it["start"]
        r0 = (s // 100) * 100
        with torch.no_grad():
            if s == r0:
                expect = torch.zeros(vec.shape[-1])
            else:
                _, h = actor(torch.tensor(ep["obs"][r0:s]).unsqueeze(0), actor.initial_hidden(1))
                expect = TD3.hidden_to_vec(h).reshape(-1)
        np.testing.assert_allclose(vec[i].numpy(), expect.numpy(), rtol=5e-3, atol=5e-3, err_msg=f"start {s}")
    # critics start from zero at b, so their burn-in is exactly the b..s run
    i = max(range(len(batch["burn"])), key=lambda j: batch["burn"][j]["start"] - batch["burn"][j]["b"])
    it = batch["burn"][i]
    l = it["start"] - it["b"]
    with torch.no_grad():
        _, _, h1, _ = critic(torch.tensor(it["obs"][:l]).unsqueeze(0), torch.tensor(it["action"][:l]).unsqueeze(0))
    np.testing.assert_allclose(TD3.hidden_to_vec(ctx["critic"][0])[i].numpy(),
                               TD3.hidden_to_vec(h1).reshape(-1).numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("core,ln", [("lstm", False), ("lstm", True), ("lru", False), ("gru", False), ("rtu", False)])
def test_td3_update_with_burn_in_runs_and_learns(core, ln, monkeypatch):
    monkeypatch.setattr(TD3, "BURN_IN", 40)
    monkeypatch.setattr(TD3, "HIDDEN_RESET_INTERVAL", 100)
    torch.manual_seed(1)
    np.random.seed(1)
    A, C = CORES[core]
    kw = {"core_ln": True} if ln else {}
    actor, actor_t = A(OBS, ACT, **kw), A(OBS, ACT, **kw)
    critic, critic_t = C(OBS, ACT, **kw), C(OBS, ACT, **kw)
    actor_t.load_state_dict(actor.state_dict()); critic_t.load_state_dict(critic.state_dict())
    for p in list(actor_t.parameters()) + list(critic_t.parameters()):
        p.requires_grad = False
    aopt = torch.optim.Adam(actor.parameters(), 3e-4)
    copt = torch.optim.Adam(critic.parameters(), 3e-4)
    rng = np.random.RandomState(2)
    demo, online = TD3.SequenceReplayBuffer(4, 60), TD3.SequenceReplayBuffer(8, 60)
    for _ in range(3):
        demo.add(_td3_ref.synthetic_episode(rng, 300))
    for _ in range(3):
        e = _td3_ref.synthetic_episode(rng, 300)
        online.add(rollout(actor, e["obs"], 100, 20).attach(e))
    before = [p.detach().clone() for p in actor.parameters()]
    update = TD3.td3_update if core == "lstm" else TD3_lru.td3_update
    for i in range(4):
        cl, al = update(actor, actor_t, critic, critic_t, aopt, copt, demo, online, i)
        assert np.isfinite(cl) and (al is None or np.isfinite(al))
    assert any(not torch.equal(b, p) for b, p in zip(before, actor.parameters()))


# ------------------------------------------------------------------ 4. LayerNorm LSTM ------------
def test_ln_lstm_interface_and_state_continuation():
    torch.manual_seed(0)
    m = LNLSTM(16, 16, 1, batch_first=True)
    ref = torch.nn.LSTM(16, 16, 1, batch_first=True)
    x = torch.randn(3, 20, 16)
    out, (h, c) = m(x)
    out_r, (h_r, c_r) = ref(x)
    assert out.shape == out_r.shape and h.shape == h_r.shape == (1, 3, 16) and c.shape == c_r.shape
    o1, s1 = m(x[:, :8])
    o2, s2 = m(x[:, 8:], s1)
    torch.testing.assert_close(torch.cat([o1, o2], 1), out)
    torch.testing.assert_close(s2[1], c)


def test_ln_lstm_output_is_bounded_even_for_extreme_inputs():
    torch.manual_seed(0)
    m = LNLSTM(16, 16, 1, batch_first=True)
    out, (h, c) = m(torch.randn(2, 200, 16) * 1e4)
    assert torch.isfinite(out).all() and torch.isfinite(c).all()
    assert out.abs().max() < 1.0
    # a hidden state far outside anything reachable cannot push the gates: h stays in (-1, 1)
    out2, _ = m(torch.randn(2, 5, 16), (torch.zeros(1, 2, 16), torch.full((1, 2, 16), 1e6)))
    assert out2.abs().max() < 1.0


def test_ln_lstm_actor_critic_wiring_and_checkpoint_detection(tmp_path):
    torch.manual_seed(3)
    a = TD3.RecurrentActor(OBS, ACT, core_ln=True)
    c = TD3.RecurrentCritic(OBS, ACT, core_ln=True)
    assert isinstance(a.lstm, LNLSTM) and isinstance(c.q1_lstm, LNLSTM) and isinstance(c.q2_lstm, LNLSTM)
    assert isinstance(TD3.RecurrentActor(OBS, ACT).lstm, torch.nn.LSTM)            # default path untouched
    x = torch.randn(2, 12, OBS)
    act, (h, cc) = a(x)
    assert act.shape == (2, 12, ACT) and h.shape == (1, 2, TD3.HIDDEN_DIM)
    q1, q2, _, _ = c(x, act)
    assert q1.shape == q2.shape == (2, 12, 1)
    assert c.q1_only(x, act).shape == (2, 12, 1)
    assert actor_io.detect_core(a.state_dict()) == "lstm_ln"
    assert actor_io.detect_core(c.state_dict()) == "lstm_ln"
    a.eval(); path = tmp_path / "actor.pth"; torch.save(a.state_dict(), path)
    loaded, core = actor_io.load_actor(str(path), device=torch.device("cpu"))
    assert core == "lstm_ln" and isinstance(loaded.lstm, LNLSTM)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x)[0], a(x)[0])
    torch.save(c.state_dict(), tmp_path / "critic.pth")
    assert actor_io.load_critic(str(tmp_path / "critic.pth"), device=torch.device("cpu"))[1] == "lstm_ln"


def test_td3_actor_controller_loads_ln_lstm(tmp_path):
    torch.manual_seed(4)
    a = TD3.RecurrentActor(OBS, ACT, core_ln=True)
    torch.save(a.state_dict(), tmp_path / "actor.pth")
    mod = load_module("td3_actor_under_test", os.path.join(PC, "controllers", "td3_actor.py"))
    ctrl = mod.Controller({"actor_path": str(tmp_path / "actor.pth"), "reset": 50})
    stir, light, frac = ctrl.act({"turbidity_ntu": 40.0, "ph": 10.0, "pump_L": 0.0, "conductivity": 5.0,
                                  "temp_c": 35.0, "lux": 30000.0})
    assert 50 <= stir <= 200 and 0 <= light <= 2000 and 0 <= frac <= 0.5


# ------------------------------------------------------------------ 5. val split ----------------
def test_val_split_is_disjoint_from_every_other_split():
    h = load_module("pc_harness_val_test", os.path.join(PC, "harness.py"))
    seeds = {n: {s for _, s in h.split_jobs(n, 2)} for n in ("search", "test", "final", "det", "val")}
    jobs = h.split_jobs("val", 2)
    assert len(jobs) == 52 and len(seeds["val"]) == 52
    assert [s for _, s in jobs[:40]] == list(range(40_000_000, 40_000_040))
    assert [s for _, s in jobs[40:]] == list(range(40_500_000, 40_500_012))
    assert all(ic >= 600 for ic, _ in jobs[40:])
    for other in ("search", "test", "final", "det"):
        assert not seeds["val"] & seeds[other], other
    # nor the demo / pilot-plant seed ranges
    assert all(s >= 40_000_000 for s in seeds["val"])
    tree = ast.parse(open(os.path.join(PC, "blackbox", "plant_trial.py"), encoding="utf-8").read())
    base = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "SEED_BASE" for t in n.targets))
    assert not seeds["val"] & set(range(base, base + 100_000 * 97 + 100_000))
    assert h.split_jobs("val", 0) == h.split_jobs("val", 2)
