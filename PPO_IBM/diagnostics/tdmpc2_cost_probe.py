"""
tdmpc2_cost_probe.py — functional smoke test + wall-clock cost measurement for the upgraded
TD-MPC2 agent (Fix v27: 3D action space, macro-timestep world model, 5-critic ensemble,
two-hot reward/value regression, project curriculum gate).

Two checks, in order — this project has a standing rule against trusting an estimate over a
measurement (the original TD-MPC2 cost claim was wrong by ~20x for exactly that reason):

  1. CORRECTNESS. TwoHotEncoder round-trip (encode a scalar, decode the encoding straight
     back) and one live agent.update() call, checked for NaN/Inf and a finite loss. This is
     read-only / no training — it exists to catch a shape or encoding bug BEFORE it burns
     hours of wall-clock in a real run.
  2. COST. Times agent.plan() and agent.update() at the file's actual configured hyperparameters
     (horizon=12, samples=64, MACRO_STEPS=50) and projects the full TOTAL_TRAINING_STEPS budget.

Usage:
    python diagnostics/tdmpc2_cost_probe.py
    python diagnostics/tdmpc2_cost_probe.py --steps 2000000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "legacy"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "environments"))

import numpy as np
import torch

from TD_MPC2 import TDMPC2Agent, TwoHotEncoder, OBS_DIM, ACTION_DIM, MACRO_STEPS, GAMMA, MAX_CELLS


def check_two_hot_roundtrip():
    print("── TwoHotEncoder round-trip ──")
    enc = TwoHotEncoder(vmin=-6.0, vmax=6.0, num_bins=101, device="cpu")
    # Values chosen to span this domain's actual measured magnitude: per-block rewards and
    # (full rationale: docs/decision_history.md#--diagnostics-tdmpc2_cost_probe-py-38)
    test_values = torch.tensor([-15.3, -1.0, -0.01, 0.0, 0.01, 0.5, 3.7, 12.9, 19.9], dtype=torch.float32)
    dist = enc.encode(test_values)
    assert torch.allclose(dist.sum(dim=-1), torch.ones(len(test_values)), atol=1e-5), \
        "two-hot rows must sum to 1.0"
    # Decode the ENCODING directly via _expected_value (skipping decode()'s softmax, which is
    # (full rationale: docs/decision_history.md#--diagnostics-tdmpc2_cost_probe-py-45)
    recovered = enc._expected_value(dist)
    max_err = (recovered - test_values).abs().max().item()
    print(f"  values:    {test_values.tolist()}")
    print(f"  recovered: {[round(v, 4) for v in recovered.tolist()]}")
    print(f"  max abs error: {max_err:.4f}  (bin width={enc.bin_width:.3f} symlog-units)")
    assert max_err < 0.5, f"two-hot round-trip error too large: {max_err:.4f}"
    print("  PASS\n")


def check_agent_smoke():
    print("── Agent smoke test (plan + update, shape/NaN check) ──")
    torch.manual_seed(0)
    np.random.seed(0)
    agent = TDMPC2Agent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cpu")

    obs = np.random.rand(OBS_DIM).astype(np.float32)
    action = agent.plan(obs, None, horizon=12, num_samples=64, num_iters=2)
    assert action.shape == (ACTION_DIM,), f"plan() returned shape {action.shape}, expected ({ACTION_DIM},)"
    assert np.all(np.isfinite(action)), "plan() returned non-finite action"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), "plan() action outside [-1,1]"
    print(f"  plan() -> action={np.round(action, 3)}  OK")

    B = 64
    batch_obs = np.random.rand(B, OBS_DIM).astype(np.float32)
    batch_mt = np.random.rand(B, OBS_DIM, 16).astype(np.float32)
    batch_actions = np.random.uniform(-1, 1, (B, ACTION_DIM)).astype(np.float32)
    # Block rewards: wider range than raw per-step rewards, matching what MACRO_STEPS-block
    # accumulation actually produces.
    batch_rewards = np.random.uniform(-5, 15, B).astype(np.float32)
    batch_dones = np.zeros(B, dtype=np.float32)
    losses = agent.update(batch_obs, batch_mt, batch_actions, batch_rewards,
                          batch_obs, batch_mt, batch_dones)
    for k, v in losses.items():
        assert np.isfinite(v), f"{k} is non-finite: {v}"
    print(f"  update() losses: {{{', '.join(f'{k}={v:.4f}' for k, v in losses.items())}}}")
    print("  PASS\n")


def measure_cost(total_steps):
    print(f"── Cost measurement at file config (horizon=12, samples=64, iters=3, "
          f"MACRO_STEPS={MACRO_STEPS}, MAX_CELLS={MAX_CELLS}) ──")
    print("  NOTE: an earlier version of this probe measured only plan()/update() in isolation")
    print("  and MISSED env.step() and the per-raw-step LMU compressor entirely — those turned")
    print("  out to dominate. All FOUR components are measured here, not assumed additive from")
    print("  a partial list.")
    torch.set_num_threads(4)
    from genetic_env import GeneticPhotobioreactorEnv
    from TD_MPC2 import ObservationBuffer

    agent = TDMPC2Agent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cpu")
    obs = np.random.rand(OBS_DIM).astype(np.float32)

    _ = agent.plan(obs, None, horizon=12, num_samples=64, num_iters=3)
    N = 6
    t0 = time.time()
    for _ in range(N):
        _ = agent.plan(obs, None, horizon=12, num_samples=64, num_iters=3)
    plan_ms = (time.time() - t0) / N * 1000
    n_plan_calls = total_steps // MACRO_STEPS
    plan_hours = plan_ms / 1000 * n_plan_calls / 3600
    print(f"  plan():       {plan_ms:7.2f} ms/call  x {n_plan_calls:,} calls  -> {plan_hours:6.2f} h")

    B = 512
    batch_obs = np.random.rand(B, OBS_DIM).astype(np.float32)
    batch_mt = np.random.rand(B, OBS_DIM, 16).astype(np.float32)
    batch_actions = np.random.uniform(-1, 1, (B, ACTION_DIM)).astype(np.float32)
    batch_rewards = np.random.uniform(-5, 15, B).astype(np.float32)
    batch_dones = np.zeros(B, dtype=np.float32)
    _ = agent.update(batch_obs, batch_mt, batch_actions, batch_rewards, batch_obs, batch_mt, batch_dones)
    N2 = 10
    t0 = time.time()
    for _ in range(N2):
        agent.update(batch_obs, batch_mt, batch_actions, batch_rewards, batch_obs, batch_mt, batch_dones)
    upd_ms = (time.time() - t0) / N2 * 1000
    n_updates = total_steps // MACRO_STEPS
    upd_hours = upd_ms / 1000 * n_updates / 3600
    print(f"  update():     {upd_ms:7.2f} ms/call  x {n_updates:,} calls  -> {upd_hours:6.2f} h")

    # env.step() — fires on EVERY raw step, not gated by MACRO_STEPS. Missed by the first
    # version of this probe; turned out to be the single largest cost component.
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=3000, difficulty=2)
    env.reset(seed=0)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    for _ in range(50):
        env.step(action)
    N3 = 300
    t0 = time.time()
    for _ in range(N3):
        env.step(action)
    env_ms = (time.time() - t0) / N3 * 1000
    env_hours = env_ms / 1000 * total_steps / 3600
    print(f"  env.step():   {env_ms:7.2f} ms/call  x {total_steps:,} calls  -> {env_hours:6.2f} h")

    # LMU compressor — also fires on EVERY raw step (fine-grained sensor history), separate
    # from the macro-block-gated plan()/update() calls. Also missed by the first probe version.
    obs_buf = ObservationBuffer(obs_dim=OBS_DIM, order=16)
    raw_obs = np.random.rand(OBS_DIM).astype(np.float32)
    obs_buf.reset(raw_obs, device="cpu")

    def _compressor_step():
        obs_tensor = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, next_m_t = agent.compressor(obs_tensor, obs_buf.get_state())
        obs_buf.set_state(next_m_t)

    for _ in range(50):
        _compressor_step()
    N4 = 2000
    t0 = time.time()
    for _ in range(N4):
        _compressor_step()
    comp_ms = (time.time() - t0) / N4 * 1000
    comp_hours = comp_ms / 1000 * total_steps / 3600
    print(f"  compressor:   {comp_ms:7.2f} ms/call  x {total_steps:,} calls  -> {comp_hours:6.2f} h")

    total_hours = plan_hours + upd_hours + env_hours + comp_hours
    print(f"\n  TOTAL projected for {total_steps:,} steps: {total_hours:.2f} h ({total_hours/24:.2f} d)")
    print(f"  (for comparison: a PPO run in this project takes ~17h)")
    return total_hours


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=int(os.environ.get("TDMPC2_STEPS", "2000000")))
    args = ap.parse_args()

    check_two_hot_roundtrip()
    check_agent_smoke()
    measure_cost(args.steps)
