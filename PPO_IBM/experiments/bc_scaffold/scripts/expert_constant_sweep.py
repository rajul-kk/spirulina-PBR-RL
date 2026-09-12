"""How sensitive is the scripted expert to its own constants?
(full rationale: docs/decision_history.md#--expert_constant_sweep-py-1)

The sharpness of each optimum is a sim2real diagnostic: a flat response means the constant
encodes a robust control principle, a sharp peak means it is tuned to this simulator's
dynamics and would mislead the BC prior on real hardware.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "environments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from genetic_env import GeneticPhotobioreactorEnv

BASE = {"setpoint": 0.015, "gain": 1.0, "cap": 0.30, "stir": 70.0, "light": 950.0}


def run(difficulty, init_cells, seed, setpoint, gain, cap, stir, light):
    np.random.seed(seed)
    env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=init_cells, difficulty=difficulty)
    env.reset(seed=seed)
    f_max = float(getattr(env, "F_MAX", 0.5))
    done, info = False, {}
    while not done:
        surplus = (float(env.od) / setpoint) - 1.0
        frac = float(np.clip(gain * surplus, 0.0, cap))
        act = np.array([np.interp(stir, [50, 200], [-1, 1]),
                        np.interp(light, [0, 2000], [-1, 1]),
                        np.interp(frac, [0, f_max], [-1, 1])], dtype=np.float32)
        _, _, term, trunc, info = env.step(act)
        done = term or trunc
    return {
        "harvest": float(info.get("cumulative_harvested_mg", 0.0)),
        "od": float(info.get("time_avg_od", 0.0)),
        "retained": env.num_active / max(float(init_cells), 1.0),
        "crashed": env.step_count < env.max_steps,
    }


def evaluate(cfg, inits, seeds, difficulty):
    rs = [run(difficulty, ic, sd, cfg["setpoint"], cfg["gain"], cfg["cap"], cfg["stir"], cfg["light"])
          for ic in inits for sd in seeds]
    return (float(np.median([r["harvest"] for r in rs])),
            float(np.median([r["od"] for r in rs])),
            float(np.median([r["retained"] for r in rs])),
            100.0 * float(np.mean([r["crashed"] for r in rs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--inits", default="380,2500")
    ap.add_argument("--seeds", default="900001,900005")
    args = ap.parse_args()
    inits = [int(x) for x in args.inits.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    GRID = [
        ("EXPERT_OD_SETPOINT", "setpoint", [0.009, 0.012, 0.015, 0.018, 0.022]),
        ("EXPERT_GAIN",        "gain",     [0.5, 1.0, 2.0, 4.0]),
        ("EXPERT_FRAC_CAP",    "cap",      [0.15, 0.30, 0.50]),
        ("stir rpm",           "stir",     [60.0, 70.0, 80.0, 140.0]),
        ("light umol",         "light",    [600.0, 950.0, 1400.0]),
    ]

    print("=" * 92)
    print(f"  EXPERT CONSTANT SENSITIVITY   D{args.difficulty}  inits={inits}  seeds={seeds}")
    print(f"  baseline: {BASE}")
    print("=" * 92)

    base_h, base_od, base_ret, base_cr = evaluate(dict(BASE), inits, seeds, args.difficulty)
    print(f"\n  BASELINE: harvest={base_h:.1f}mg  od={base_od:.4f}  retained={100*base_ret:.0f}%  crash={base_cr:.0f}%\n")

    for label, key, vals in GRID:
        print(f"  {label}  (baseline {BASE[key]})")
        print(f"    {'value':>10} {'harvest':>9} {'vs base':>9} {'od':>8} {'retained':>9} {'crash':>6}")
        rows = []
        for v in vals:
            cfg = dict(BASE); cfg[key] = v
            h, od, ret, cr = evaluate(cfg, inits, seeds, args.difficulty)
            rows.append((v, h))
            mark = "  <-- baseline" if v == BASE[key] else ""
            print(f"    {v:10.4g} {h:9.1f} {100*(h/max(base_h,1e-9)-1):+8.1f}% {od:8.4f} "
                  f"{100*ret:8.0f}% {cr:5.0f}%{mark}")
        hs = [h for _, h in rows]
        best_v = rows[int(np.argmax(hs))][0]
        spread = (max(hs) - min(hs)) / max(np.median(hs), 1e-9)
        verdict = "FLAT (robust prior)" if spread < 0.15 else ("MODERATE" if spread < 0.5 else "SHARP (sim-specific)")
        print(f"    -> best={best_v:g}  spread={100*spread:.0f}% of median  {verdict}\n")

    print("  A SHARP optimum means the constant is fitted to this simulator and the BC prior it")
    print("  generates would carry that bias onto hardware. FLAT means it encodes a principle.")


if __name__ == "__main__":
    sys.exit(main())
