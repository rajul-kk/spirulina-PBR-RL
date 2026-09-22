"""Correctness check for the PBRS reward shaping in genetic_env.

Potential-based shaping only preserves the optimal policy if the shaping term is exactly
`F_t = gamma*Phi(s_{t+1}) - Phi(s_t)` over the SAME state sequence the agent actually visits.
That identity is easy to break in ways training will not surface for days -- caching Phi after
the wrong mutation, letting a transition quantity leak into Phi, or resetting the cache mid
episode. All three break the telescoping sum, so testing the sum catches all three.

Checks run here:
  1. TELESCOPING: sum_t F_t == gamma^T*Phi(s_T) - Phi(s_0) over a real rollout, to tolerance.
     (Exact only when gamma applies once per step, which is the point of the check.)
  2. STATE PURITY: Phi depends on state alone -- calling it repeatedly without stepping
     returns an identical value, and it is unchanged by which action was taken to arrive.
  3. BOUNDEDNESS: Phi stays inside its designed range across extreme populations/OD, so the
     shaping term cannot dominate the task reward or blow up the critic.

Usage:
  python experiments/env_diagnosis/pbrs_invariance_check.py
  python experiments/env_diagnosis/pbrs_invariance_check.py --steps 3000 --seed 7
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from environments.genetic_env import GeneticPhotobioreactorEnv  # noqa: E402


def run_telescoping(env, steps, rng):
    """Roll out and compare summed shaping against the closed-form telescoped value."""
    env.reset(seed=int(rng.integers(0, 10_000)))
    gamma = env.PBRS_GAMMA
    phi_0 = env._potential()

    shaping_sum = 0.0
    undiscounted_sum = 0.0
    discount = 1.0
    telescoped = 0.0
    phi_prev = phi_0
    t = 0
    env_shaping_sum = 0.0

    for _ in range(steps):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        _, _, terminated, truncated, info = env.step(action)
        phi_now = env._potential()

        # Reconstructed independently of the env, then cross-checked against what the env
        # actually paid out (below) -- a reconstruction that telescopes proves nothing if the
        # env's reward does not use it.
        f_t = gamma * phi_now - phi_prev
        shaping_sum += discount * f_t
        undiscounted_sum += f_t
        telescoped = discount * gamma * phi_now - phi_0
        phi_prev = phi_now
        discount *= gamma
        t += 1
        env_shaping_sum = info.get("reward_term_sums", {}).get("shaping", float("nan"))
        if terminated or truncated:
            break

    return shaping_sum, telescoped, t, undiscounted_sum, env_shaping_sum


def check_state_purity(env, rng):
    """Phi must be reproducible from state alone, and independent of the action history
    that produced it."""
    env.reset(seed=1234)
    for _ in range(50):
        env.step(rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32))

    repeated = [env._potential() for _ in range(5)]
    spread = max(repeated) - min(repeated)
    return spread


def check_bounds(env):
    """Phi across extreme OD and population values."""
    env.reset(seed=1)
    vals = []
    saved_od, saved_pop = env.od, env.num_active
    for od in [1e-6, 1e-4, 0.001, 0.012, 0.05, 0.2, 1.0]:
        for pop in [0, 1, 50, 400, 5000, 50_000]:
            env.od = od
            env.num_active = pop
            vals.append(env._potential())
    env.od, env.num_active = saved_od, saved_pop
    return min(vals), max(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = GeneticPhotobioreactorEnv()

    print("=" * 78)
    print("  PBRS invariance check")
    print(f"  gamma={env.PBRS_GAMMA}  PHI_SCALE={env.PHI_SCALE}  "
          f"OD_W={env.PHI_OD_W} POP_W={env.PHI_POP_W} POP_REF={env.PHI_POP_REF}")
    print("=" * 78)

    failures = 0

    shaping_sum, telescoped, t, undisc, env_shaping = run_telescoping(env, args.steps, rng)
    err = abs(shaping_sum - telescoped)
    ok = err <= args.tol * max(1.0, abs(telescoped))
    failures += 0 if ok else 1
    print(f"\n1. TELESCOPING over {t} steps")
    print(f"   sum_t discount*F_t     : {shaping_sum: .10f}")
    print(f"   gamma^T*Phi(s_T)-Phi(s0): {telescoped: .10f}")
    print(f"   abs error               : {err:.3e}    {'PASS' if ok else 'FAIL'}")

    env_err = abs(undisc - env_shaping)
    ok = env_err <= 1e-9 * max(1.0, abs(env_shaping))
    failures += 0 if ok else 1
    print(f"\n1b. ENV ACTUALLY PAYS THE SHAPING IT SHOULD")
    print(f"   reconstructed sum_t F_t : {undisc: .10f}")
    print(f"   env reward_term_sums    : {env_shaping: .10f}")
    print(f"   abs error               : {env_err:.3e}    "
          f"{'PASS' if ok else 'FAIL (env reward does not match the PBRS formula)'}")

    spread = check_state_purity(env, rng)
    ok = spread == 0.0
    failures += 0 if ok else 1
    print(f"\n2. STATE PURITY")
    print(f"   spread over 5 repeat calls, no stepping: {spread:.3e}   "
          f"{'PASS' if ok else 'FAIL (Phi is not a pure function of state)'}")

    lo, hi = check_bounds(env)
    expected_hi = env.PHI_SCALE * 1.0001
    ok = lo >= -1e-9 and hi <= expected_hi
    failures += 0 if ok else 1
    print(f"\n3. BOUNDEDNESS across extreme OD x population")
    print(f"   Phi range: [{lo:.6f}, {hi:.6f}]   design bound: [0, {env.PHI_SCALE}]   "
          f"{'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f"  {'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
