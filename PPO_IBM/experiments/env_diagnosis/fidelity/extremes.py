"""Drive the env through extreme operating regimes and print the range of every state variable,
NaN/negative flags, crash step, harvest, and sensor saturation.

  python experiments/env_diagnosis/fidelity/extremes.py [scenario-substring ...]
"""
import sys
import time

import numpy as np

from common import act, const, expert, fmt_ext, run


def square(period_steps, stir=65.0, hi=2000.0):
    a_on, a_off = act(stir, hi, 0.0), act(stir, 0.0, 0.0)
    return lambda env, t: a_on if (t // (period_steps // 2)) % 2 == 0 else a_off


def max_harvest(light=1400.0, stir=65.0):
    a = act(stir, light, 0.5)
    return lambda env, t: a


def harvest_then_idle(n_events=3, light=1400.0, stir=65.0):
    a_h, a_i = act(stir, light, 0.5), act(stir, light, 0.0)
    return lambda env, t: a_h if t < n_events * 600 else a_i


SCENARIOS = [
    # name, policy, init_cells, difficulty
    ("dark 6 days (light 0, 65 rpm)", const(65, 0), 700, 2),
    ("full light 2000 continuous, no harvest", const(65, 2000), 700, 2),
    ("full light 2000 continuous, no harvest D0", const(65, 2000), 700, 0),
    ("square wave 0/2000, 24 h period", square(1200), 700, 2),
    ("square wave 0/2000, 1 h period", square(50), 700, 2),
    ("stir 50, 1400 umol, no harvest", const(50, 1400), 700, 2),
    ("stir 200, 1400 umol, no harvest", const(200, 1400), 700, 2),
    ("stir 50, expert harvest", expert(stir=50), 700, 2),
    ("stir 200, expert harvest", expert(stir=200), 700, 2),
    ("no harvest ever, expert light/stir", const(65, 1400), 700, 2),
    ("max harvest 0.5 every event", max_harvest(), 700, 2),
    ("max harvest 0.5, 3000 cells", max_harvest(), 3000, 2),
    ("harvest 0.5 x3 then idle", harvest_then_idle(), 700, 2),
    ("tiny inoculum 30, expert", expert(), 30, 2),
    ("tiny inoculum 30, no harvest", const(65, 1400), 30, 2),
    ("max inoculum 7500, no harvest", const(65, 1400), 7500, 2),
    ("max inoculum 7500, expert", expert(), 7500, 2),
    ("expert D0", expert(), 700, 0),
    ("expert D2", expert(), 700, 2),
    ("light 0 + stir 200 (cold/heat check)", const(200, 0), 700, 2),
]

KEYS = ("od", "cells", "max_agent_mass", "temp", "ph", "do2", "dissolved_co2", "conductivity", "salt",
        "n_pool", "p_pool", "alkalinity", "dic", "clump", "pigment", "membrane_integrity", "kLa",
        "obs_turb", "obs_lux", "obs_cond")

if __name__ == "__main__":
    sel = sys.argv[1:]
    for name, pol, ic, d in SCENARIOS:
        if sel and not any(s in name for s in sel):
            continue
        t0 = time.time()
        r = run(pol, init_cells=ic, difficulty=d, seed=7)
        f = r["final"]
        L = r["ledger"]
        print(f"\n=== {name}  (init {ic}, D{d})  {time.time() - t0:.0f}s")
        print(f"  steps {r['steps']} crash_t {r['crash_t']} harvest {r['harvest_mg']:.0f} mg  "
              f"final od {f['od']:.3f} ({f['od'] * 300:.0f} mg/L) cells {f['cells']:.0f} "
              f"T {f['temp']:.2f} pH {f['ph']:.2f} clump {f['clump']:.2f}  NaN/neg flags {r['n_bad']} {r['bad'][:3]}")
        print(f"  lysed {L['lysed']:.0f} mg, agent-steps at 5e8 mass cap {L['cap_agent_steps']}, "
              f"turbidity >=999 NTU {r['turb_sat_frac']:.0%} of steps")
        print("  " + fmt_ext(r["ext"], KEYS))
        sys.stdout.flush()
