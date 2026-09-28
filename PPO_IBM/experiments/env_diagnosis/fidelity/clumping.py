"""Flocculation drift: mean clump size vs stir speed and density, and what clumping costs in growth
(the same run with aggregation switched off).

  python experiments/env_diagnosis/fidelity/clumping.py [hours]
"""
import sys

import numpy as np

from common import LedgerEnv, const, expert, run


class NoClumpEnv(LedgerEnv):
    def _update_flocculation(self, stir_rpm):
        pass


def summary(pol, init, env_cls, hours):
    r = run(pol, init_cells=init, difficulty=2, seed=9, steps=int(hours / 0.02), env_cls=env_cls,
            record_every=600)
    clumps = [row["clump"] for row in r["trace"]]
    return r["final"]["od"] * 300, r["harvest_mg"], clumps, r["final"]["do2"]


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 72.0
    for label, pol, init in (("expert harvest @ 65 rpm, 700 cells", expert(stir=65), 700),
                             ("no harvest @ 65 rpm, 700 cells", const(65, 1400), 700),
                             ("no harvest @ 100 rpm, 700 cells", const(100, 1400), 700),
                             ("no harvest @ 65 rpm, 7500 cells", const(65, 1400), 7500),
                             ("no harvest @ 120 rpm, 7500 cells", const(120, 1400), 7500)):
        X, H, clumps, do2 = summary(pol, init, LedgerEnv, hours)
        X0, H0, _, _ = summary(pol, init, NoClumpEnv, hours)
        print(f"\n=== {label}, {hours:.0f} h")
        print(f"   clump every 12 h: {' '.join(f'{c:.1f}' for c in clumps)}")
        print(f"   end X {X:.0f} mg/L (no clumping: {X0:.0f}), harvest {H:.0f} mg (no clumping: {H0:.0f}), DO {do2:.1f}")
        sys.stdout.flush()
