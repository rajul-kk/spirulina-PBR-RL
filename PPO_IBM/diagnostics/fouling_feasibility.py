"""
fouling_feasibility.py — would enabling REAL biofouling make D2 unreachable, and does it
make stir an interesting control lever?

Two questions, both of which must be answered before recommending that the light-path
fouling coefficient be raised from its (inert) historical 0.0002:

  Q1 FEASIBILITY. Fouling attenuates all light channels by exp(-fouling_factor), which
     throttles growth, which lowers time_avg_od — the exact criterion that gates D2
     (>=0.011). If active fouling puts D2 out of reach even for the best known controller,
     enabling it would set the agent an impossible target.

  Q2 INTERESTINGNESS. Fouling rate scales with (1 - stir/200), so stir becomes a real
     mitigation. But higher stir also costs yield. If the best stir under fouling differs
     from the best stir without it, fouling turns stir from a near-irrelevant dial into a
     genuine trade-off — and because fouling ACCUMULATES, the optimal stir becomes
     time-varying, which a constant-stir controller cannot exploit but a recurrent policy
     can. That would be a regime where RL should beat the scripted expert.

Read-only probe: drives the env with the scripted OD-feedback harvest law (same one
bc_pretrain.py clones) at a range of fixed stir settings, with fouling off vs on.
"""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--diagnostics-fouling_feasibility-py-24)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environments"))
from genetic_env import GeneticPhotobioreactorEnv

EXPERT_OD_SETPOINT = 0.015
EXPERT_GAIN = 1.0
EXPERT_FRAC_CAP = 0.30
LIGHT = 950.0

STIRS = [60.0, 80.0, 100.0, 130.0, 160.0]
SEEDS = [0, 1, 2]
D1_GATE = (60.0, 30.0, 0.008)
D2_GATE = (90.0, 50.0, 0.011)


def run(stir, coef, seed, difficulty=2):
    np.random.seed(seed)
    GeneticPhotobioreactorEnv.LIGHT_FOULING_COEF = coef
    env = GeneticPhotobioreactorEnv(max_cells=7_500, initial_cells=300,
                                    difficulty=difficulty, enable_fouling=True)
    env.reset(seed=seed)
    f_max = float(env.F_MAX)
    done, info = False, {}
    while not done:
        surplus = (float(env.od) / EXPERT_OD_SETPOINT) - 1.0
        frac = float(np.clip(EXPERT_GAIN * surplus, 0.0, EXPERT_FRAC_CAP))
        act = np.array([
            np.interp(stir, [50, 200], [-1, 1]),
            np.interp(LIGHT, [0, 2000], [-1, 1]),
            np.interp(frac, [0, f_max], [-1, 1]),
        ], dtype=np.float32)
        _o, _r, term, trunc, info = env.step(act)
        done = term or trunc
    return (float(info.get("cumulative_harvested_mg", 0.0)),
            float(info.get("time_avg_od", 0.0)),
            float(env.fouling_factor))


def main():
    print("Scripted OD-feedback expert, light=950, D2 physics, 3 seeds per cell\n")
    for coef, label in ((0.0002, "fouling OFF (historical, inert)"),
                        (0.075, "fouling ON (realistic for this OD scale)")):
        print(f"  {label}")
        print(f"  {'stir':>5} {'harvest_mg':>11} {'time_avg_od':>12} {'foul':>7} {'light_thru':>11}  D1  D2")
        best = (None, -1.0)
        for stir in STIRS:
            hs, ods, fs = [], [], []
            for s in SEEDS:
                h, od, f = run(stir, coef, s)
                hs.append(h); ods.append(od); fs.append(f)
            mh, mo, mf = float(np.median(hs)), float(np.median(ods)), float(np.median(fs))
            thru = float(np.exp(-mf))
            d1 = "PASS" if (mh >= D1_GATE[0] and mo >= D1_GATE[2]) else "fail"
            d2 = "PASS" if (mh >= D2_GATE[0] and mo >= D2_GATE[2]) else "fail"
            print(f"  {stir:5.0f} {mh:11.1f} {mo:12.4f} {mf:7.4f} {thru*100:10.1f}%  {d1}  {d2}")
            if mo > best[1]:
                best = (stir, mo)
        print(f"    -> best stir for time_avg_od: {best[0]:.0f}rpm ({best[1]:.4f})\n")

    print("READ:")
    print("  Q1 if 'fouling ON' has NO row passing D2, enabling it makes the target unreachable.")
    print("  Q2 if the best stir SHIFTS between the two blocks, fouling makes stir a real,")
    print("     state-dependent lever — the kind a recurrent policy can exploit and a")
    print("     constant-stir expert cannot.")


if __name__ == "__main__":
    main()
