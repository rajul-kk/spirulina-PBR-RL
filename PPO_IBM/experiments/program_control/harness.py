"""Evaluation harness for controller *programs*: hand-written, CMA-ES-tuned or LLM-written.

A controller program is a Python file defining

    class Controller:
        def __init__(self, params=None): ...
        def act(self, obs: dict) -> tuple[float, float, float]:
            # returns (stir_rpm 50-200, light_umol 0-2000, harvest_frac 0-0.5)

`obs` holds only what a real reactor's sensors report (see SENSOR_DOC), plus the controller's
own clock. The harness converts the physical action to the env's [-1, 1] action space, so a
program never touches the env object. `privileged=True` adds the simulator's true OD, and is
used only for the oracle reference (the TD3 demo expert reads env.od directly).

Splits (all seeds disjoint):
  search  D2, 9 main + 3 high-pop episodes, seeds 3,000,000+  -- the only split search sees
  test    D2, the td3_held_out_sweep protocol (base seed 1000: 40 main + 12 high-pop)
  det     the curriculum's fixed DET_EVAL_SET (9 instances)

  python experiments/program_control/harness.py <controller.py> [--split test] [--difficulty 2]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

MAX_CELLS = 7500      # TD3.MAX_CELLS; not imported so workers skip loading torch
ADVERSARIAL_MAX = 80  # at/below this the episode is scored on survival, not yield
EPISODE_WALL_S = 300  # a program slower than this per episode forfeits the rest of it

SENSOR_DOC = """\
obs keys (all sensor readings carry noise; at D1+ also drift, lag and a pH bias):
  turbidity_ntu   nephelometer, ~250 NTU per OD unit at the start of a batch; reads low as cells
                  clump over days (up to ~30%), saturates at 1000
  ph              pH probe (Zarrouk medium, pH-stat CO2 keeps it near 10)
  pump_L          cumulative harvest pump volume, litres
  conductivity    uS/cm
  temp_c          broth temperature, C (thermostat setpoint 35; strong light heats the tank)
  lux             BH1750 reading of the LED panel, ~30 lux per umol/m2/s (echoes your light)
  t               step index (1 step = 0.02 h = 72 s; episode = 7200 steps = 144 h)
  dt_h            0.02
Harvest: every 600 steps (12 h) the env removes the MEAN of the harvest_frac values you output
over the preceding 600 steps (cap 0.5) and refills with fresh medium. Nutrients are dosed
automatically. OD target ~0.75 (1 OD unit = 300 mg/L dry weight)."""


def _ctrl_obs(obs, t, env=None):
    d = {"turbidity_ntu": float(obs[0]), "ph": float(obs[1]), "pump_L": float(obs[2]),
         "conductivity": float(obs[3]), "temp_c": float(obs[4]), "lux": float(obs[5]),
         "t": int(t), "dt_h": 0.02}
    if env is not None:
        d["true_od"] = float(env.od)
    return d


def load_controller_class(path):
    spec = importlib.util.spec_from_file_location(f"ctrl_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Controller


def to_env_action(stir, light, frac, f_max):
    stir = float(np.clip(stir, 50.0, 200.0))
    light = float(np.clip(light, 0.0, 2000.0))
    frac = float(np.clip(frac, 0.0, f_max))
    return np.array([np.interp(stir, [50, 200], [-1, 1]),
                     np.interp(light, [0, 2000], [-1, 1]),
                     np.interp(frac, [0, f_max], [-1, 1])], dtype=np.float32)


def run_episode(job):
    """job: (controller_path, params, difficulty, init_cells, seed, privileged, trace_every)."""
    from genetic_env import GeneticPhotobioreactorEnv
    path, params, difficulty, init_cells, seed, privileged, trace_every = job
    np.random.seed(seed)   # env.reset(seed) alone doesn't seed strain randomisation
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=init_cells, difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    f_max = float(env.F_MAX)
    trace, done, t, info, err = [], False, 0, {}, None
    try:   # a program that fails to load or construct forfeits the episode, like one failing in act()
        Controller = load_controller_class(path)
        ctrl = Controller(params) if params is not None else Controller()
    except Exception as e:
        err = f"{type(e).__name__} at load: {e}"
        done = True
    t_start = time.time()
    while not done:
        if t % 500 == 0 and time.time() - t_start > EPISODE_WALL_S:
            err = f"TimeoutError: episode exceeded {EPISODE_WALL_S}s"
            break
        try:
            stir, light, frac = ctrl.act(_ctrl_obs(obs, t, env if privileged else None))
            if not all(np.isfinite([stir, light, frac])):
                raise ValueError(f"non-finite action {(stir, light, frac)}")
        except Exception as e:  # a broken program forfeits the rest of the episode
            err = f"{type(e).__name__}: {e}"
            break
        action = to_env_action(stir, light, frac, f_max)
        if trace_every and t % trace_every == 0:
            trace.append({"t_h": round(t * 0.02, 1), "turb": round(float(obs[0]), 1),
                          "temp_obs": round(float(obs[4]), 2), "ph": round(float(obs[1]), 2),
                          "stir": round(float(stir), 1), "light": round(float(light), 1),
                          "frac": round(float(frac), 3), "true_od_over_target": round(float(env.od) / float(env.OD_TARGET), 3),
                          "true_temp": round(float(env.temp), 2), "cells": int(env.num_active),
                          "harvested_mg": round(float(env.cumulative_harvested_mg), 1)})
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        t += 1
    return {"seed": seed, "init_cells": init_cells, "steps": t, "crashed": t < env.max_steps,
            "error": err, "harvested_mg": float(env.cumulative_harvested_mg),
            "time_avg_od": float(info.get("time_avg_od", 0.0)), "trace": trace}


def split_jobs(split, difficulty):
    """(init_cells, seed) pairs. 'test' reproduces td3_held_out_sweep.py exactly."""
    if split == "det":
        from curriculum_schedule import DET_EVAL_SET
        return list(DET_EVAL_SET)

    def main_block(base, n):
        rng = np.random.RandomState(base)
        out = []
        for i in range(n):
            if rng.rand() < 0.10:
                ic = int(rng.uniform(30, 80))
            else:
                ic = int(np.exp(rng.uniform(np.log(100), np.log(400))))
            out.append((ic, base + i))
        return out

    def high_block(base, n):
        rng = np.random.RandomState(base + 500_000)
        return [(int(np.exp(rng.uniform(np.log(600), np.log(5000)))), base + 500_000 + i) for i in range(n)]

    if split == "search":
        return main_block(3_000_000, 9) + high_block(3_000_000, 3)
    if split == "test":
        return main_block(1000, 40) + high_block(1000, 12)
    raise ValueError(split)


def summarise(results):
    yielding = [r for r in results if r["init_cells"] > ADVERSARIAL_MAX]
    h = np.array([r["harvested_mg"] for r in yielding]) if yielding else np.zeros(1)
    crash = float(np.mean([r["crashed"] for r in results]))
    s = {"n": len(results), "median_mg": float(np.median(h)), "p25_mg": float(np.percentile(h, 25)),
         "mean_mg": float(np.mean(h)), "crash_rate": crash,
         "median_time_avg_od": float(np.median([r["time_avg_od"] for r in yielding])) if yielding else 0.0,
         "errors": sum(1 for r in results if r["error"])}
    # One scalar for search: rewards typical and bad-case yield, charges 10 g per 100% crash.
    s["fitness"] = 0.5 * s["median_mg"] + 0.5 * s["p25_mg"] - 10_000.0 * crash
    return s


def evaluate(path, split="search", difficulty=2, params=None, privileged=False, workers=2,
             trace_every=0):
    path = os.path.abspath(path)
    jobs = [(path, params, difficulty, ic, seed, privileged, trace_every)
            for ic, seed in split_jobs(split, difficulty)]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(run_episode, jobs)
    else:
        results = [run_episode(j) for j in jobs]
    return summarise(results), results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controller")
    ap.add_argument("--split", default="search", choices=["search", "test", "det"])
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--params", default=None, help="JSON dict passed to Controller(params)")
    ap.add_argument("--privileged", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None, help="write summary + per-episode results as JSON")
    args = ap.parse_args()
    params = json.loads(args.params) if args.params else None
    t0 = time.time()
    s, res = evaluate(args.controller, args.split, args.difficulty, params, args.privileged, args.workers)
    print(f"{os.path.basename(args.controller)} split={args.split} D{args.difficulty} n={s['n']}: "
          f"median {s['median_mg']:.0f} mg, p25 {s['p25_mg']:.0f} mg, crash {100*s['crash_rate']:.0f}%, "
          f"time-avg od {s['median_time_avg_od']:.2f}, errors {s['errors']}, fitness {s['fitness']:.0f} "
          f"[{time.time()-t0:.0f}s]")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": s, "episodes": [{k: v for k, v in r.items() if k != "trace"} for r in res]},
                      f, indent=1)


if __name__ == "__main__":
    main()
