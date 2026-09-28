"""Pilot-plant trials for the black-box controller writer.

Runs NEW batches (never repeated) of the reactor under a controller program and returns what a
plant would record: the sensor log, the actions taken, the biomass weighed at each 12 h harvest,
and an offline lab dry-weight assay taken just before each harvest (+-5% assay error). The
simulator itself is not exposed.

  python plant_trial.py <controller.py> [--n 4] [--inoculum CELLS] [--label name]

Prints a per-batch summary; writes an hourly CSV log per batch to trials/<batch_id>.csv.
Budget: 300 batches in total (trials/budget.json).
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from harness import MAX_CELLS, _ctrl_obs, load_controller_class, to_env_action  # noqa: E402

BUDGET = 300
TRIALS = os.path.join(HERE, "trials")
SEED_BASE = 5_000_000          # disjoint from the search (3M), test (1000) and calibration (7M) seeds
ASSAY_ERR = 0.05
DIFFICULTY = 2                 # plant conditions: noisy, drifting, lagging sensors; actuator error


def sample_inoculum(rng):
    u = rng.rand()
    if u < 0.10:
        return int(rng.uniform(30, 80))
    if u < 0.30:
        return int(np.exp(rng.uniform(np.log(600), np.log(5000))))
    return int(np.exp(rng.uniform(np.log(100), np.log(400))))


def run_batch(job):
    from genetic_env import GeneticPhotobioreactorEnv
    path, batch_id, seed, inoculum = job
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=MAX_CELLS, initial_cells=inoculum, difficulty=DIFFICULTY)
    obs, _ = env.reset(seed=seed)
    f_max = float(env.F_MAX)
    arng = np.random.RandomState(seed + 17)
    rows, events, err, t = [], [], None, 0
    try:
        ctrl = load_controller_class(path)()
    except Exception as e:
        err = f"{type(e).__name__} at load: {e}"
    done = err is not None
    while not done:
        o = _ctrl_obs(obs, t)
        try:
            stir, light, frac = ctrl.act(o)
            if not all(np.isfinite([stir, light, frac])):
                raise ValueError(f"non-finite action {(stir, light, frac)}")
        except Exception as e:
            err = f"{type(e).__name__} at t={t}: {e}"
            break
        is_event = t > 0 and t % 600 == 0
        if is_event:   # lab assay just before the harvest is applied
            assay = float(env.od) * 300.0 * (1.0 + ASSAY_ERR * arng.randn())
            before = float(env.cumulative_harvested_mg)
        obs, _, terminated, truncated, _ = env.step(to_env_action(stir, light, frac, f_max))
        if is_event:
            events.append({"hour": round(t * 0.02, 1), "lab_dry_weight_mg_per_L": round(assay, 1),
                           "harvested_mg": round(float(env.cumulative_harvested_mg) - before, 1)})
        if t % 50 == 0:
            rows.append([round(t * 0.02, 2)] + [round(float(v), 3) for v in obs[:6]]
                        + [round(float(stir), 2), round(float(light), 1), round(float(frac), 4)])
        t += 1
        done = terminated or truncated
    with open(os.path.join(TRIALS, f"{batch_id}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hour", "turbidity_ntu", "ph", "pump_L", "conductivity", "temp_c", "lux",
                    "stir_rpm", "light_umol", "harvest_frac"])
        w.writerows(rows)
    return {"batch": batch_id, "inoculum": inoculum, "hours_run": round(t * 0.02, 1),
            "culture_lost": t < 7200, "error": err,
            "total_harvested_mg": round(float(env.cumulative_harvested_mg), 1), "harvests": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controller")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--inoculum", type=int, default=None, help="fix the starting culture size (cells)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    os.makedirs(TRIALS, exist_ok=True)
    bpath = os.path.join(TRIALS, "budget.json")
    used = json.load(open(bpath))["used"] if os.path.exists(bpath) else 0
    n = min(args.n, BUDGET - used)
    if n <= 0:
        print(f"budget exhausted ({used}/{BUDGET} batches used)")
        return
    rng = np.random.RandomState(SEED_BASE + used)
    jobs = []
    for i in range(n):
        k = used + i
        inoc = args.inoculum if args.inoculum else sample_inoculum(rng)
        inoc = int(np.clip(inoc, 30, MAX_CELLS))
        jobs.append((os.path.abspath(args.controller), f"b{k:03d}{('_' + args.label) if args.label else ''}",
                     SEED_BASE + k, inoc))
    json.dump({"used": used + n}, open(bpath, "w"))
    with Pool(2) as pool:
        res = pool.map(run_batch, jobs)
    with open(os.path.join(TRIALS, "results.jsonl"), "a") as f:
        for r in res:
            f.write(json.dumps({**r, "controller": os.path.basename(args.controller)}) + "\n")
    for r in res:
        h = " ".join(f"{e['harvested_mg']:.0f}" for e in r["harvests"])
        print(f"{r['batch']}: inoculum {r['inoculum']}, harvested {r['total_harvested_mg']:.0f} mg"
              f"{', CULTURE LOST at %.1f h' % r['hours_run'] if r['culture_lost'] else ''}"
              f"{', ERROR ' + r['error'] if r['error'] else ''} | per-harvest mg: {h}")
    print(f"budget: {used + n}/{BUDGET} batches used; logs in trials/")


if __name__ == "__main__":
    main()
