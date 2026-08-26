"""
reward_ab.py — head-to-head per-term reward comparison: trained policy vs scripted expert,
on IDENTICAL episodes (same seed, same initial_cells, same difficulty).

WHY: v17 (BC warm start) inverted the expert's phase structure — it harvests 0.25-0.30
early and declines to ~0.18, whereas the expert it was cloned from harvests ~0 early and
ramps up. The expert scores better on the curriculum gates, so the question is whether the
REWARD also prefers the expert. If the reward prefers v17's behaviour, the reward is the
problem. If the reward prefers the expert and PPO drifted anyway, it is an optimisation
problem.

The standing lesson from the v8 reweighting mistake applies: measure per-term totals before
changing any weight. This script produces that measurement.

Usage:
    python reward_ab.py --model model_data/archive_v17_bc_warmstart_D2_8M/recurrent_ppo_genetic_ibm \
                        --norm  model_data/archive_v17_bc_warmstart_D2_8M/recurrent_vec_normalize.pkl \
                        --n 8 --difficulty 2
"""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-reward_ab-py-21)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environments"))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from genetic_env import GeneticPhotobioreactorEnv

# Must match bc_pretrain.py's expert exactly.
EXPERT_OD_SETPOINT = 0.015
EXPERT_GAIN = 1.0
EXPERT_FRAC_CAP = 0.30
EXPERT_STIR = 70.0
EXPERT_LIGHT = 950.0

TERMS = ("od", "biomass", "od_delta", "harvest")


def make_vec(norm_path, difficulty, init_cells):
    def _make():
        env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells,
                                        difficulty=difficulty)
        return Monitor(env)
    base = DummyVecEnv([_make])
    vec = VecNormalize.load(norm_path, venv=base)
    vec.training = False
    vec.norm_reward = False
    return vec


def raw_env_of(vec):
    e = vec
    if hasattr(e, "venv"):
        e = e.venv
    e = e.envs[0]
    if hasattr(e, "env"):
        e = e.env
    return e


def encode(stir, light, frac, f_max):
    return np.array([
        np.interp(stir, [50, 200], [-1, 1]),
        np.interp(light, [0, 2000], [-1, 1]),
        np.interp(frac, [0, f_max], [-1, 1]),
    ], dtype=np.float32)


def run(vec, policy_fn, seed):
    np.random.seed(seed)
    obs = vec.reset()
    raw = raw_env_of(vec)
    state = {"lstm": None, "starts": np.ones((1,), dtype=bool)}
    done, info, total, fracs = False, {}, 0.0, []
    f_max = float(getattr(raw, "F_MAX", 0.5))
    while not done:
        action = policy_fn(obs, raw, state, f_max)
        obs, reward, done_vec, info_list = vec.step(action)
        total += float(reward[0])
        done = bool(done_vec[0])
        info = info_list[0] if info_list else {}
        fracs.append(float(np.interp(action[0][2], [-1, 1], [0.0, f_max])))
    return {
        "terms": dict(info.get("reward_term_sums", {})),
        "total": total,
        "harvested": float(info.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info.get("time_avg_od", 0.0)),
        "frac_first": float(np.mean(fracs[:600])),
        "frac_last": float(np.mean(fracs[-600:])),
        "frac_mean": float(np.mean(fracs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=3000)
    args = ap.parse_args()

    rng = np.random.RandomState(args.base_seed)
    inits = [int(np.exp(rng.uniform(np.log(100), np.log(400)))) for _ in range(args.n)]

    probe = make_vec(args.norm, args.difficulty, inits[0])
    model = RecurrentPPO.load(args.model, env=probe, device="cpu",
                              custom_objects={"observation_space": probe.observation_space})

    def model_policy(obs, raw, state, f_max):
        act, state["lstm"] = model.predict(obs, state=state["lstm"],
                                           episode_start=state["starts"], deterministic=True)
        state["starts"] = np.zeros((1,), dtype=bool)
        return act

    def expert_policy(obs, raw, state, f_max):
        surplus = (float(getattr(raw, "od", 0.0)) / EXPERT_OD_SETPOINT) - 1.0
        frac = float(np.clip(EXPERT_GAIN * surplus, 0.0, EXPERT_FRAC_CAP))
        return encode(EXPERT_STIR, EXPERT_LIGHT, frac, f_max).reshape(1, -1)

    rows = {"v17": [], "expert": []}
    for i, init in enumerate(inits):
        seed = args.base_seed + i
        for name, fn in (("v17", model_policy), ("expert", expert_policy)):
            vec = make_vec(args.norm, args.difficulty, init)
            if name == "v17":
                state_reset = True  # fresh lstm per episode via model_policy's state dict
            rows[name].append(run(vec, fn, seed))
            vec.close()
        a, b = rows["v17"][-1], rows["expert"][-1]
        print(f"  ep {i+1}/{args.n} init={init:>4}  "
              f"v17: rew={a['total']:8.1f} harv={a['harvested']:6.1f} od={a['time_avg_od']:.4f} "
              f"frac {a['frac_first']:.2f}->{a['frac_last']:.2f}  ||  "
              f"expert: rew={b['total']:8.1f} harv={b['harvested']:6.1f} od={b['time_avg_od']:.4f} "
              f"frac {b['frac_first']:.2f}->{b['frac_last']:.2f}")

    print(f"\n{'='*88}")
    print(f"  PER-TERM REWARD TOTALS (mean over {args.n} identical episodes, D{args.difficulty})")
    print(f"{'='*88}")
    print(f"  {'term':<12} {'v17':>12} {'expert':>12} {'expert-v17':>12}   who does the reward prefer?")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for t in TERMS:
        av = float(np.mean([r["terms"].get(t, 0.0) for r in rows["v17"]]))
        bv = float(np.mean([r["terms"].get(t, 0.0) for r in rows["expert"]]))
        print(f"  {t:<12} {av:>12.1f} {bv:>12.1f} {bv-av:>+12.1f}   "
              f"{'expert' if bv > av else 'v17'}")
    at = float(np.mean([r["total"] for r in rows["v17"]]))
    bt = float(np.mean([r["total"] for r in rows["expert"]]))
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'TOTAL':<12} {at:>12.1f} {bt:>12.1f} {bt-at:>+12.1f}   "
          f"{'expert' if bt > at else 'v17'}")
    print(f"\n  outcome    v17: harvest {np.mean([r['harvested'] for r in rows['v17']]):6.1f}mg  "
          f"time_avg_od {np.mean([r['time_avg_od'] for r in rows['v17']]):.4f}")
    print(f"  outcome expert: harvest {np.mean([r['harvested'] for r in rows['expert']]):6.1f}mg  "
          f"time_avg_od {np.mean([r['time_avg_od'] for r in rows['expert']]):.4f}")
    print(f"{'='*88}\n")
    if bt > at:
        print("  READ: reward PREFERS the expert -> reward ranking is correct, PPO drifted away")
        print("        from a higher-reward policy. That is an optimisation problem.")
    else:
        print("  READ: reward PREFERS v17's behaviour -> the reward function itself pays for")
        print("        early over-harvesting. That is a reward-structure problem; the term with")
        print("        the largest positive 'v17 - expert' gap above is the one responsible.")


if __name__ == "__main__":
    main()
