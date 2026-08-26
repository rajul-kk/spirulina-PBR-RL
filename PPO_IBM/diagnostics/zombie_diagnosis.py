"""
zombie_diagnosis.py — diagnoses the "zombie" failure mode found by reward_breakdown.py:
episodes that never hard-crash (terminate early) but spend an extended stretch with
OD < 0.001, racking up heavy washout penalty and dragging deterministic reward deeply
negative even though the same checkpoint reports 0% crash_rate in the curriculum gate.

For each episode, tracks per-step OD/action and reports:
  - init_cells, difficulty
  - whether the episode ever entered a "zombie" stretch (>=20 consecutive steps od<0.001)
  - zombie onset step, zombie duration (steps), whether it recovered before episode end
  - mean action (stir/light/harvest_frac) in the 200 steps before zombie onset vs during
  - final reward_term_sums, cumulative_harvested_mg, time_avg_od, crashed
"""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-zombie_diagnosis-py-15)
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

ZOMBIE_OD_THRESHOLD = 0.001
ZOMBIE_MIN_STEPS = 20


def sample_init_cells(rng):
    if rng.rand() < 0.10:
        return int(rng.uniform(30, 80))
    return int(np.exp(rng.uniform(np.log(100), np.log(400))))


def run_episode(model, norm_path, difficulty, init_cells, seed):
    def _make():
        env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
        return Monitor(env)
    base = DummyVecEnv([_make])
    vec_env = VecNormalize.load(norm_path, venv=base)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()
    lstm_states = None
    ep_starts = np.ones((1,), dtype=bool)
    done = False
    step = 0

    od_trace = []
    action_trace = []  # (stir, light, harvest_frac) each step
    terms = {}
    info_final = {}

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
        ep_starts = np.zeros((1,), dtype=bool)
        obs, reward, done_vec, info = vec_env.step(action)
        done = bool(done_vec[0])
        step += 1
        od_trace.append(float(info[0].get("od", 0.0)))
        action_trace.append(np.array(action[0], dtype=np.float32))
        if done:
            terms = info[0].get("reward_term_sums", {})
            info_final = info[0]

    od_arr = np.array(od_trace)
    act_arr = np.array(action_trace)  # shape (steps, action_dim)

    # Detect the first zombie stretch: >=ZOMBIE_MIN_STEPS consecutive steps with od<threshold.
    below = od_arr < ZOMBIE_OD_THRESHOLD
    zombie_onset = None
    zombie_duration = 0
    zombie_recovered = None
    run_start = None
    run_len = 0
    for i, b in enumerate(below):
        if b:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_len >= ZOMBIE_MIN_STEPS and zombie_onset is None:
                zombie_onset = run_start
                zombie_duration = run_len
                zombie_recovered = True
            run_start = None
            run_len = 0
    # trailing run (never recovered by episode end)
    if run_len >= ZOMBIE_MIN_STEPS and zombie_onset is None:
        zombie_onset = run_start
        zombie_duration = run_len
        zombie_recovered = False

    total_zombie_steps = int(np.sum(below))

    pre_action_mean = None
    zombie_action_mean = None
    if zombie_onset is not None:
        pre_lo = max(0, zombie_onset - 200)
        pre_action_mean = act_arr[pre_lo:zombie_onset].mean(axis=0) if zombie_onset > pre_lo else None
        zombie_action_mean = act_arr[zombie_onset:zombie_onset + zombie_duration].mean(axis=0)

    return {
        "seed": seed,
        "init_cells": init_cells,
        "difficulty": difficulty,
        "steps": step,
        "crashed": step < 7200,
        "min_od": float(od_arr.min()),
        "max_od": float(od_arr.max()),
        "zombie_onset": zombie_onset,
        "zombie_duration": zombie_duration,
        "zombie_recovered": zombie_recovered,
        "total_zombie_steps": total_zombie_steps,
        "pre_action_mean": pre_action_mean,
        "zombie_action_mean": zombie_action_mean,
        "overall_action_mean": act_arr.mean(axis=0),
        "reward_terms": terms,
        "total_reward": float(sum(terms.values())),
        "cumulative_harvested_mg": float(info_final.get("cumulative_harvested_mg", 0.0)),
        "time_avg_od": float(info_final.get("time_avg_od", 0.0)),
    }


def fmt_action(a):
    if a is None:
        return "n/a"
    return "stir={:.2f} light={:.2f} harvest={:.2f}".format(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--norm", type=str, default="model_data/recurrent_vec_normalize.pkl")
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--base-seed", type=int, default=3000)
    args = ap.parse_args()

    print(f"Loading model {args.model} ...")
    model = RecurrentPPO.load(args.model)

    rng = np.random.RandomState(args.base_seed)
    results = []
    for i in range(args.n):
        seed = args.base_seed + i
        init_cells = sample_init_cells(rng)
        np.random.seed(seed)
        rec = run_episode(model, args.norm, args.difficulty, init_cells, seed)
        results.append(rec)
        tag = "ZOMBIE" if rec["zombie_onset"] is not None else "healthy"
        print(f"  [{i+1}/{args.n}] seed={seed} init={init_cells:5d} {tag:7s} "
              f"onset={str(rec['zombie_onset']):>5s} dur={rec['zombie_duration']:4d} "
              f"recovered={rec['zombie_recovered']} min_od={rec['min_od']:.5f} "
              f"total_reward={rec['total_reward']:+7.1f}")

    print(f"\n{'='*100}")
    print("  ZOMBIE DIAGNOSIS SUMMARY")
    print(f"{'='*100}")
    zombies = [r for r in results if r["zombie_onset"] is not None]
    healthy = [r for r in results if r["zombie_onset"] is None]
    print(f"  zombie episodes : {len(zombies)}/{len(results)}")
    print(f"  healthy episodes: {len(healthy)}/{len(results)}")

    if zombies:
        z_init = np.array([r["init_cells"] for r in zombies])
        h_init = np.array([r["init_cells"] for r in healthy]) if healthy else np.array([])
        print(f"\n  init_cells — zombie: min={z_init.min()} max={z_init.max()} median={np.median(z_init):.0f}")
        if healthy:
            print(f"  init_cells — healthy: min={h_init.min()} max={h_init.max()} median={np.median(h_init):.0f}")

        onsets = np.array([r["zombie_onset"] for r in zombies])
        durations = np.array([r["zombie_duration"] for r in zombies])
        recovered = np.array([r["zombie_recovered"] for r in zombies])
        print(f"\n  zombie onset step : min={onsets.min()} max={onsets.max()} median={np.median(onsets):.0f}")
        print(f"  zombie duration   : min={durations.min()} max={durations.max()} median={np.median(durations):.0f}")
        print(f"  recovered before episode end: {int(recovered.sum())}/{len(zombies)}")

        # Action comparison: mean action just before zombie onset vs during zombie
        pre_actions = np.array([r["pre_action_mean"] for r in zombies if r["pre_action_mean"] is not None])
        zom_actions = np.array([r["zombie_action_mean"] for r in zombies])
        if len(pre_actions):
            print(f"\n  mean action (200 steps BEFORE zombie onset): {fmt_action(pre_actions.mean(axis=0))}")
        print(f"  mean action (DURING zombie stretch)         : {fmt_action(zom_actions.mean(axis=0))}")
        if healthy:
            h_actions = np.array([r["overall_action_mean"] for r in healthy])
            print(f"  mean action (healthy episodes, whole episode): {fmt_action(h_actions.mean(axis=0))}")

        z_reward = np.mean([r["total_reward"] for r in zombies])
        h_reward = np.mean([r["total_reward"] for r in healthy]) if healthy else float("nan")
        print(f"\n  mean total_reward — zombie: {z_reward:+.2f}   healthy: {h_reward:+.2f}")


if __name__ == "__main__":
    main()
