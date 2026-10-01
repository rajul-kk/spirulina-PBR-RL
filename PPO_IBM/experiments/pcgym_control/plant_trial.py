"""Pilot-plant trial tool for the PC-Gym writer runs (docs/reports/pcgym_protocol.md).

Runs a controller for n batches on fresh scenarios, logs every step to a CSV per batch and
appends one summary line per batch to results.jsonl. Budget: 300 batches per run
(runs/<run>/trials/budget.json). Each run draws its own scenarios (a seed block per run name),
so repeats never share data. --privileged adds the plant's true state and the unmeasured
disturbances to the logs (white-box writers only).

  python plant_trial.py --run bb1 runs/bb1/work/ctrl.py --n 6 --label first
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import check_source, run_episode  # noqa: E402

BUDGET = 300
SEED_BASE = 5_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controller")
    ap.add_argument("--run", required=True, help="run name (own budget, scenarios and logs)")
    ap.add_argument("--task", default="cstr")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--label", default="trial")
    ap.add_argument("--privileged", action="store_true")
    args = ap.parse_args()

    trials = os.path.join(HERE, "runs", args.run, "trials")
    os.makedirs(trials, exist_ok=True)
    bpath = os.path.join(trials, "budget.json")
    used = json.load(open(bpath))["used"] if os.path.exists(bpath) else 0
    n = min(args.n, BUDGET - used)
    if n <= 0:
        print(f"budget exhausted ({used}/{BUDGET} batches used)")
        return
    check_source(args.controller)
    seed_base = SEED_BASE + 100_000 * (1 + sum(map(ord, args.run)) % 97)
    for k in range(used, used + n):
        summary, trace = run_episode((os.path.abspath(args.controller), args.task, seed_base + k,
                                      None, args.privileged, True))
        name = f"b{k:03d}_{args.label}"
        with open(os.path.join(trials, name + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0]))
            w.writeheader()
            w.writerows(trace)
        rec = {"batch": name, "controller": os.path.basename(args.controller), "cost": summary["cost"],
               "runaway": summary["runaway"], "t_max": summary["t_max"], "error": summary["error"]}
        with open(os.path.join(trials, "results.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        json.dump({"used": k + 1}, open(bpath, "w"))
        print(f"{name}: cost {summary['cost']:.3f}  max T {summary['t_max']:.1f} K"
              f"{'  RUNAWAY' if summary['runaway'] else ''}{'  ERROR ' + summary['error'] if summary['error'] else ''}")
    print(f"budget: {used + n}/{BUDGET} batches used; logs in runs/{args.run}/trials/")


if __name__ == "__main__":
    main()
